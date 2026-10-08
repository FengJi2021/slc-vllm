# slc-vllm (`dvllm`)

A small, readable LLM inference engine written in PyTorch, built for learning
how engines like [vLLM](https://github.com/vllm-project/vllm) work: paged KV
cache, a block manager with prefix hashing, a prefill/decode scheduler, and a
hand-written Qwen3 model. It runs on **CUDA**, **Apple Silicon (MPS)** and
**CPU**, and ships an **OpenAI-compatible HTTP server** (`slc-server`) so you can
use it from any OpenAI SDK, LiteLLM, or agent framework.

Currently supported model: **Qwen3** (tested with
[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)).

```
dvllm/
  engine/    LLMEngine, Scheduler, BlockManager (paged KV cache), RunModel
  layers/    attention (flash-attn / SDPA fallback), rotary, sampler, ...
  models/    Qwen3 implementations
  server/    OpenAI-compatible FastAPI server (slc-server)
test/        example, throughput benchmark, server smoke test
tools/       debugging helpers
```

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- git
- macOS on Apple Silicon (MPS), Linux + NVIDIA GPU (CUDA), or CPU

## Setup

```shell
git clone git@github.com:FengJi2021/slc-vllm.git
cd slc-vllm
uv sync
```

### Download the model

```shell
uv run python download_model.py      # -> ~/huggingface/Qwen3-0.6B
```

or with the Hugging Face CLI:

```shell
uvx --from huggingface_hub huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B
```

## Run the OpenAI-compatible server

```shell
uv run slc-server \
  --model ~/huggingface/Qwen3-0.6B \
  --host 0.0.0.0 --port 8000 \
  --served-model-name qwen3-0.6b
```

The device is picked automatically (CUDA > MPS > CPU; override with
`--device`). The default `--host` is `127.0.0.1`; use `0.0.0.0` so other
machines or Docker containers can connect. Startup takes about 10 s for the
0.6B model.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | required | Local Hugging Face model directory |
| `--host` / `--port` | `127.0.0.1` / `8000` | Bind address |
| `--served-model-name` | model dir name | Model id returned by `/v1/models` |
| `--device` | `auto` | `auto`, `cuda`, `mps` or `cpu` |
| `--max-model-len` | engine default | Max prompt + completion tokens (MPS caps it at 2048) |
| `--kv-blocks` | enough for one full context | KV cache blocks (256 tokens each) |
| `--default-max-tokens` | `1024` | Completion limit when a request sets none |
| `--api-key` | none (env `SLC_API_KEY`) | Require `Authorization: Bearer <key>` |
| `--log-level` | `info` | Each request logs its queue time, TTFT and decode tok/s |

### Endpoints

- `GET /health`: status, device, and queue length
- `GET /v1/models`
- `POST /v1/chat/completions`: non-streaming, or streaming with `stream: true`
  (Server-Sent Events in the OpenAI chunk format, ending with `data: [DONE]`)

Supported request fields: `messages` (string or text-part content),
`max_tokens` / `max_completion_tokens`, `temperature` (0 means greedy), `top_p`,
`top_k`, `presence_penalty`, `frequency_penalty`, `stop`, `stream`,
`stream_options.include_usage`, `tools` / `tool_choice: "none"`, and
`chat_template_kwargs`. The server ignores fields it doesn't know (`user`,
`seed`, `response_format`, `n`, ...) and doesn't reject them.

**Qwen3 thinking.** Thinking is **off** by default (`enable_thinking=False`
in the chat template). To turn it on for a request, send
`"chat_template_kwargs": {"enable_thinking": true}`. The `<think>...</think>`
text is returned in `message.reasoning_content` (or `delta.reasoning_content`
when streaming), separate from `content`.

**Tools.** Tool definitions go through the Qwen3 chat template, and any
`<tool_call>` blocks in the output come back as OpenAI `tool_calls` with
`finish_reason: "tool_calls"`. When a request includes tools, the tool calls
arrive in one chunk at the end of the stream. A 0.6B model calls tools
unreliably.

### curl

```shell
curl http://localhost:8000/health
curl http://localhost:8000/v1/models

# non-streaming
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "qwen3-0.6b",
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "What is the capital of France?"}
        ],
        "max_tokens": 64
      }'

# streaming
curl -N http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "qwen3-0.6b",
        "messages": [{"role": "user", "content": "Write a haiku about the sea."}],
        "stream": true,
        "stream_options": {"include_usage": true}
      }'
```

### OpenAI Python client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[{"role": "user", "content": "Name three primary colors."}],
    max_tokens=64,
)
print(resp.choices[0].message.content, resp.usage)

for chunk in client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[{"role": "user", "content": "Explain a KV cache briefly."}],
    stream=True,
):
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

# opt in to Qwen3 thinking
resp = client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[{"role": "user", "content": "What is 17 + 25?"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)
print(resp.choices[0].message.reasoning_content)
```

`test/server_smoke.py` runs these checks plus a stop-sequence test, a tools
test, and a concurrency test against a running server:

```shell
uv run python test/server_smoke.py --base-url http://localhost:8000/v1
```

### Behind LiteLLM or an OpenAI-compatible agent

Any client that accepts a custom OpenAI base URL works. Set:

- base URL: `http://<host>:8000/v1`
- API key: any non-empty string, or the value of `--api-key`
- model: the `--served-model-name`, e.g. `qwen3-0.6b`

LiteLLM proxy `config.yaml`:

```yaml
model_list:
  - model_name: qwen3-0.6b
    litellm_params:
      model: openai/qwen3-0.6b          # "openai/" = generic OpenAI-compatible
      api_base: http://host.docker.internal:8000/v1
      api_key: EMPTY
```

LiteLLM Python SDK:

```python
import litellm
litellm.completion(
    model="openai/qwen3-0.6b",
    api_base="http://localhost:8000/v1",
    api_key="EMPTY",
    messages=[{"role": "user", "content": "hi"}],
)
```

**From Docker on macOS** (Docker Desktop / OrbStack): run the server
**natively on the host** with `--host 0.0.0.0`, then point the container at
`http://host.docker.internal:8000/v1`.

## Limitations

- **One request at a time.** The engine's batched path isn't correct for
  serving yet: multi-sequence prefill samples only one token, the built-in
  sampler applies one temperature to the whole batch, and prefix-cache hits
  are ignored by the non-CUDA attention. The server therefore runs one
  sequence at a time on a dedicated worker thread. Concurrent requests wait
  in a FIFO queue and don't fail, so total throughput does not grow with
  concurrency. The server also clears prefix-cache hashes before each
  request, so a shared system prompt is computed again every time.
- **Context length**: 2048 tokens on MPS (prompt + completion). Longer
  prompts get a `400 context_length_exceeded` error. A larger `max_tokens` is
  reduced to fit.
- **Apple GPU only when run natively.** MPS (Metal) is only available to a
  process running directly on macOS. Docker containers on a Mac (Docker
  Desktop, OrbStack, Colima) have no Metal access, so inside a container the
  engine would fall back to CPU. Run the server on the host and connect from
  the container.
- **Prefill is slow on MPS/CPU.** The fallback attention writes the KV cache
  token by token in Python, so TTFT grows with prompt length. Rough numbers on
  a Mac mini M4 (16 GB) with Qwen3-0.6B at concurrency 1: TTFT about 0.1–0.3 s
  for short prompts and about 3.8 s for a 1.5k-token prompt; decode about
  21–24 tokens/s.
- Not implemented: `n > 1`, `logprobs`, `seed`, `response_format`/JSON mode,
  `/v1/completions`, and `/v1/embeddings`.

## Development

```shell
uv sync --all-extras --group dev
source .venv/bin/activate
```

### Get model params

```shell
./tools/get_params.py --model ModelPath (ex:~/huggingface/Qwen3-0.6B/model.safetensors)
```

### Example run

This checks that inference runs correctly (e.g. on macOS / MPS):

```shell
./test/run.sh
```

### Throughput benchmark

Measures prefill and decode throughput over several batch sizes (average,
min and max over several trials):

```shell
python test/benchmark.py                 # default sweep on auto device
python test/benchmark.py --device mps
python test/benchmark.py --batch-sizes 1,2,4,8 --trials 3 --max-tokens 128
```

On MPS the default KV cache only reserves a few blocks. Use `--kv-blocks` to
raise it so larger batches fit.

### Debugging

See [DEBUG_README.md](DEBUG_README.md) for layer-by-layer debugging
(`DV_DEBUG_LAYERS`) and [tools/](tools/) for helpers.
