# Benchmarks

Where dvllm stands against the official [mlx-lm](https://github.com/ml-explore/mlx-lm)
on the same Mac and model. Update this page when a change moves a headline
number, in the same PR.

## How to measure

Warm runs only (exclude model load and the first iteration), same model
weights and dtype, nothing else heavy running on the machine.

**dvllm, engine only** (prefill and decode tok/s over batch sizes):

```shell
uv run python test/benchmark.py --device mps --batch-sizes 1,2,4 --trials 3 --max-tokens 128
```

**dvllm, through the server** (TTFT and decode tok/s for one request, as a
client sees it):

```shell
uv run slc-server --model ~/huggingface/Qwen3-0.6B &
uv run python test/server_smoke.py      # the [stream] line prints ttft and decode tok/s
```

**mlx-lm reference** (in its own environment, not this repo's):

```shell
uvx --from mlx-lm mlx_lm.generate --model Qwen/Qwen3-0.6B \
  --prompt "Explain what a KV cache is in about 100 words." --max-tokens 160
# prints prompt tok/s, generation tok/s and peak memory
```

mlx-lm uses its own bf16 conversion of the weights; note the exact model id.

## Results

Qwen3-0.6B, bf16, batch size 1. Fill a row per machine; keep old rows when a
number moves so the trend stays visible.

| Date | Commit | Machine | Engine | Prompt tokens | TTFT (s) | Prefill tok/s | Decode tok/s | Peak mem |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-08 | 9cdee86 | Mac mini M4 16 GB | dvllm (server, MPS) | short | 0.1–0.3 | | 21–24 | |
| 2026-10-08 | 9cdee86 | Mac mini M4 16 GB | dvllm (server, MPS) | ~1500 | ~3.8 | | 21–24 | |
| | | Mac mini M4 16 GB | mlx-lm | | | | | |
