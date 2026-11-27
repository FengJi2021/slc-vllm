#!/usr/bin/env python3
"""Greedy-decode parity: dvllm vs Hugging Face transformers on the same model.

A perf change must not change what the model says. This runs a few prompts
greedily through both, and fails if dvllm's tokens diverge from the
reference before --min-match tokens (low-precision kernels may legitimately
drift late in a long completion, so we don't demand an exact match of all of
them).

    uv run python test/parity.py --model ~/huggingface/Qwen3-0.6B --device cpu
"""
import argparse
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dvllm import LLM, SParams
from dvllm.engine.seq import Sequence
from dvllm.server.worker import RequestSampler, SamplingConfig

PROMPTS = [
    "What is the capital of France?",
    "List the first ten prime numbers.",
    "Explain what a KV cache is in one sentence.",
]


def reference(model_dir, prompt_ids, max_tokens, dtype):
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype)
    model.eval()
    out = []
    with torch.inference_mode():
        for ids in prompt_ids:
            x = torch.tensor([ids])
            y = model.generate(
                x,
                attention_mask=torch.ones_like(x),
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=model.config.eos_token_id,
            )
            out.append(y[0, len(ids):].tolist())
    del model
    return out


def dvllm_greedy(model_dir, prompt_ids, max_tokens, device):
    llm = LLM(model_dir, enforce_eager=True, tensor_parallel_size=1,
              device_type=device, max_model_len=1024, num_kvcache_blocks=8)
    sampler = RequestSampler()
    sampler.cfg = SamplingConfig(temperature=0.0)
    llm.model_runner.sampler = sampler
    out = []
    # One sequence at a time, as the server runs it (batched prefill is not
    # correct yet; see dvllm/server/worker.py).
    for ids in prompt_ids:
        llm.scheduler.block_manager.hash_to_block_id.clear()
        seq = Sequence(ids, SParams(temperature=1e-5, max_tokens=max_tokens))
        sampler.seq = seq
        llm.scheduler.add(seq)
        while not seq.is_finished:
            llm.step()
        out.append(seq.completion_token_ids)
    return out, llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--min-match", type=int, default=16,
                    help="tokens that must match the reference exactly")
    args = ap.parse_args()
    model_dir = os.path.expanduser(args.model)

    tok = AutoTokenizer.from_pretrained(model_dir)
    prompt_ids = [
        tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True,
                                add_generation_prompt=True, enable_thinking=False)
        for p in PROMPTS
    ]

    got, llm = dvllm_greedy(model_dir, prompt_ids, args.max_tokens, args.device)
    dtype = llm.model_runner.model.lm_head.weight.dtype
    want = reference(model_dir, prompt_ids, args.max_tokens, dtype)

    failed = False
    for p, g, w in zip(PROMPTS, got, want):
        n = next((i for i, (a, b) in enumerate(zip(g, w)) if a != b), min(len(g), len(w)))
        ok = n >= min(args.min_match, len(w))
        failed |= not ok
        print(f"[{'ok' if ok else 'FAIL'}] {n}/{len(w)} tokens match  {p!r}")
        if not ok:
            print(f"    dvllm: {tok.decode(g)!r}")
            print(f"    ref:   {tok.decode(w)!r}")
    print(f"dtype={dtype} device={args.device}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
