# Server

`slc-server`, an OpenAI-compatible HTTP server over the engine. Source:
[`dvllm/server/`](../../dvllm/server/). User-facing flags and request fields
are documented in the [README](../../README.md#run-the-openai-compatible-server).

## Parts

- **[cli.py](../../dvllm/server/cli.py)** parses flags, resolves the device
  (CUDA > MPS > CPU), sizes `max_model_len` and the KV cache for it, builds
  the `LLMEngine`, and runs uvicorn.
- **[app.py](../../dvllm/server/app.py)** (`create_app`) serves
  `/health`, `/v1/models` and `/v1/chat/completions`. It validates the
  request, applies the chat template (thinking off unless
  `chat_template_kwargs.enable_thinking`), rejects prompts over the
  context with `400 context_length_exceeded`, and shapes the reply:
  `split_reasoning` moves `<think>` text to `reasoning_content`,
  `extract_tool_calls` turns `<tool_call>` blocks into `tool_calls`.
  Streaming is SSE in OpenAI chunk format ending with `data: [DONE]`.
  Optional bearer auth (`--api-key` / `SLC_API_KEY`).
- **[worker.py](../../dvllm/server/worker.py)** (`EngineWorker`) owns the
  engine on one thread and runs one `Job` at a time from a FIFO queue.
  Each `engine.step()` yields one token; the worker decodes, checks stop
  tokens and stop strings, holds back partial UTF-8 and stop-string
  prefixes, and emits `text` / `done` / `error` events to the request's
  asyncio queue. A disconnected client sets `cancelled`, and the sequence is
  aborted and its blocks freed.

## Why one sequence at a time

The engine's batched path isn't correct for serving yet (see the module
docstring of `worker.py` and [LESSONS.md](../../LESSONS.md#batching)).
Concurrent requests queue rather than fail, so throughput doesn't grow with
concurrency. Lifting this is the top throughput item in the
[roadmap](../roadmap.md); it needs batched prefill that samples every
sequence, per-sequence sampling params, and prefix-cache-aware MPS
attention, each proven by parity at batch size > 1.

## Metrics

Each request logs `queue`, `ttft`, `total` and decode tok/s at `info`.

## Verification

Start the server, then `test/server_smoke.py` (non-stream, stream with TTFT,
stop string, unknown fields, thinking, tools, concurrency, `/health`).
