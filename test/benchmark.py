#!/usr/bin/env python3
"""
Throughput benchmark for dvllm on MPS / CPU / CUDA.

Measures prefill and decode throughput across several batch sizes, running
multiple trials per configuration and reporting the average (plus min/max).

Usage:
    python test/benchmark.py                 # default sweep on auto device
    python test/benchmark.py --device mps
    python test/benchmark.py --batch-sizes 1,2,4,8 --trials 3 --max-tokens 128

Notes on MPS:
  * KV cache is block-paged. Each sequence needs ceil(prompt_len / block_size)
    blocks. The default MPS config only reserves 4 blocks, so large batches of
    even short prompts will not fit. Use --kv-blocks to raise it (memory
    permitting) so batching can actually be exercised.
"""
import argparse
import gc
import logging
import os
import statistics
import time

import torch

from dvllm import LLM, SParams
from dvllm.engine.seq import Sequence


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def build_prompt(tokenizer, target_len: int) -> str:
    """Build a chat prompt whose tokenized length is close to target_len."""
    filler = "Please write a detailed technical explanation. "
    body = ""
    while len(tokenizer.encode(body)) < target_len:
        body += filler
    return (
        "<|im_start|>user\n"
        + body.strip()
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def run_trial(llm, prompts, max_tokens, device):
    """Run one generate() call, return (prefill_tok, decode_tok, prefill_s, decode_s)."""
    sp = SParams(temperature=0.7, max_tokens=max_tokens)

    n_prompt_tokens = sum(len(llm.tokenizer.encode(p)) for p in prompts)

    # We instrument the engine step() timing by wrapping the scheduler step.
    # Simpler: time first step (prefill) separately from the rest (decode).
    from dvllm.engine.run_model import RunModel

    orig_run = RunModel.run
    timings = {"prefill_s": 0.0, "decode_s": 0.0, "decode_steps": 0, "decode_tok": 0}

    def timed_run(self, seqs, is_prefill, dump_path=None):
        sync(device)
        t = time.perf_counter()
        r = orig_run(self, seqs, is_prefill, dump_path)
        sync(device)
        dt = time.perf_counter() - t
        if is_prefill:
            timings["prefill_s"] += dt
        else:
            timings["decode_s"] += dt
            timings["decode_steps"] += 1
            timings["decode_tok"] += len(seqs)
        return r

    RunModel.run = timed_run
    try:
        outs = llm.generate(prompts, sp, use_tqdm=False)
    finally:
        RunModel.run = orig_run

    decode_tok = sum(len(o["token_ids"]) for o in outs)
    return {
        "n_prompt_tokens": n_prompt_tokens,
        "prefill_s": timings["prefill_s"],
        "decode_s": timings["decode_s"],
        "decode_tok": decode_tok,
        "decode_steps": timings["decode_steps"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    ap.add_argument("--batch-sizes", default="1,2,4,8")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--warmup-trials", type=int, default=1)
    ap.add_argument("--prompt-len", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument(
        "--kv-blocks",
        type=int,
        default=0,
        help="Override num_kvcache_blocks (0 = engine default). Raise for batching.",
    )
    args = ap.parse_args()

    logging.disable(logging.CRITICAL)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    device = pick_device(args.device)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]

    print(f"# dvllm throughput benchmark")
    print(f"# model      : {args.model}")
    print(f"# device     : {device}")
    print(f"# prompt_len : ~{args.prompt_len} tokens")
    print(f"# max_tokens : {args.max_tokens}")
    print(f"# trials     : {args.trials} (+{args.warmup_trials} warmup)")
    print()

    # Optionally patch the KV-cache block count so larger batches fit.
    if args.kv_blocks > 0:
        import dvllm.engine.run_model as RM

        orig_alloc = RM.RunModel.allocate_kv_cache

        def patched_alloc(self):
            orig_config_blocks = self.config.num_kvcache_blocks
            orig_alloc(self)  # allocates with engine default first
            # Re-allocate with the requested block count.
            self.config.num_kvcache_blocks = args.kv_blocks
            # rebuild cache tensor
            hf = self.config.hf_config
            num_kv_heads = hf.num_key_value_heads // self.world_size
            head_dim = getattr(
                hf, "head_dim", hf.hidden_size // hf.num_attention_heads
            )
            self.kv_cache = torch.empty(
                2,
                hf.num_hidden_layers,
                args.kv_blocks,
                self.block_size,
                num_kv_heads,
                head_dim,
                device=self.device,
                dtype=hf.torch_dtype,
            )
            layer_id = 0
            for module in self.model.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = self.kv_cache[0, layer_id]
                    module.v_cache = self.kv_cache[1, layer_id]
                    layer_id += 1

        RM.RunModel.allocate_kv_cache = patched_alloc

    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        device_type=device,
    )

    base_prompt = build_prompt(llm.tokenizer, args.prompt_len)
    actual_prompt_len = len(llm.tokenizer.encode(base_prompt))
    print(f"# actual prompt length: {actual_prompt_len} tokens\n")

    header = (
        f"{'batch':>5} | {'prefill tok/s':>13} | {'decode tok/s':>12} | "
        f"{'decode ms/step':>14} | {'total tok/s':>11}"
    )
    print(header)
    print("-" * len(header))

    for bs in batch_sizes:
        prompts = [base_prompt] * bs

        # warmup
        for _ in range(args.warmup_trials):
            Sequence.counter = __import__("itertools").count()
            run_trial(llm, prompts, args.max_tokens, device)
            gc.collect()

        prefill_tps = []
        decode_tps = []
        decode_ms = []
        total_tps = []
        for _ in range(args.trials):
            Sequence.counter = __import__("itertools").count()
            r = run_trial(llm, prompts, args.max_tokens, device)
            gc.collect()

            if r["prefill_s"] > 0:
                prefill_tps.append(r["n_prompt_tokens"] / r["prefill_s"])
            if r["decode_s"] > 0:
                decode_tps.append(r["decode_tok"] / r["decode_s"])
                decode_ms.append(r["decode_s"] / max(1, r["decode_steps"]) * 1000)
            total = r["n_prompt_tokens"] + r["decode_tok"]
            total_tps.append(total / (r["prefill_s"] + r["decode_s"]))

        def fmt(vals):
            if not vals:
                return "n/a"
            m = statistics.mean(vals)
            return f"{m:8.1f}"

        print(
            f"{bs:>5} | {fmt(prefill_tps):>13} | {fmt(decode_tps):>12} | "
            f"{fmt(decode_ms):>14} | {fmt(total_tps):>11}"
        )

    print()
    print("# decode tok/s is aggregate across the batch (sum of all streams).")


if __name__ == "__main__":
    main()
