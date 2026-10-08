"""``slc-server`` entry point: load the model and serve the OpenAI-compatible API."""

from __future__ import annotations

import os

# Must be set before torch initialises MPS: lets unsupported ops fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import logging  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="slc-server", description="OpenAI-compatible HTTP server for dvllm"
    )
    p.add_argument("--model", required=True, help="Path to a local HF model dir (e.g. ~/huggingface/Qwen3-0.6B)")
    p.add_argument("--host", default="127.0.0.1", help="Bind address (use 0.0.0.0 to accept LAN / Docker traffic)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--served-model-name", default=None, help="Model id reported by /v1/models (default: model dir name)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--max-model-len", type=int, default=None, help="Max prompt+completion tokens (MPS caps this at 2048)")
    p.add_argument("--kv-blocks", type=int, default=0, help="KV cache blocks of 256 tokens (0 = enough for max-model-len on MPS/CPU, auto on CUDA)")
    p.add_argument("--default-max-tokens", type=int, default=1024, help="Completion limit when the request sets none")
    p.add_argument("--api-key", default=os.environ.get("SLC_API_KEY"), help="Require 'Authorization: Bearer <key>' (env SLC_API_KEY)")
    p.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    return p.parse_args(argv)


def resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # The engine logs per-step debug info; keep it quiet unless asked.
    if args.log_level != "debug":
        logging.getLogger("dvllm.engine").setLevel(logging.WARNING)
        logging.getLogger("dvllm.layers").setLevel(logging.WARNING)
        logging.getLogger("dvllm.models").setLevel(logging.WARNING)

    import uvicorn

    from dvllm import LLM
    from dvllm.server.app import create_app
    from dvllm.server.worker import EngineWorker

    model = os.path.abspath(os.path.expanduser(args.model))
    if not os.path.isdir(model):
        raise SystemExit(f"model directory not found: {model}")
    device = resolve_device(args.device)
    name = args.served_model_name or os.path.basename(model.rstrip("/"))

    kwargs = dict(enforce_eager=True, tensor_parallel_size=1, device_type=device)
    if args.max_model_len:
        kwargs["max_model_len"] = args.max_model_len
        kwargs["max_num_batched_tokens"] = max(16384 if device == "cuda" else 2048, args.max_model_len)
    if args.kv_blocks > 0:
        kwargs["num_kvcache_blocks"] = args.kv_blocks
    elif device in ("mps", "cpu"):
        # One sequence runs at a time, so enough blocks for one full context.
        max_len = min(args.max_model_len or 4096, 2048 if device == "mps" else 4096)
        kwargs["num_kvcache_blocks"] = (max_len + 255) // 256

    log = logging.getLogger("dvllm.server")
    log.info("loading %s on %s ...", model, device)
    engine = LLM(model, **kwargs)
    cfg = engine.model_runner.config
    kv_tokens = cfg.num_kvcache_blocks * cfg.kvcache_block_size
    max_context = min(cfg.max_model_len, kv_tokens, cfg.max_num_batched_tokens)
    log.info(
        "model ready: device=%s max_context=%d kv_blocks=%d served_model_name=%s",
        engine.model_runner.device,
        max_context,
        cfg.num_kvcache_blocks,
        name,
    )

    worker = EngineWorker(engine)
    app = create_app(
        worker,
        served_model_name=name,
        max_context=max_context,
        default_max_tokens=args.default_max_tokens,
        api_key=args.api_key,
        device=str(engine.model_runner.device),
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
