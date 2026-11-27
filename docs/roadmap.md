# Roadmap

Known gaps, in rough priority order. Each item lands only with parity
passing and benchmark numbers ([benchmarks.md](benchmarks.md)). Remove an
item when it's done; its record is the PR and the benchmark row.

## Latency (MPS)

1. **Vectorize the KV cache write.** MPS/CPU writes K/V one token at a time
   in Python (`layers/attention.py`), so TTFT grows with prompt length
   (~3.8 s for 1.5k tokens on an M4). Use `index_copy_` / scatter on
   `slot_mapping`.
2. **Cut per-step Python overhead in decode.** Profile one decode step
   (`prepare_decode`, context setup, sampler, `.tolist()` syncs) and remove
   host work and device syncs from the hot path.
3. **Metal kernels where SDPA isn't enough**, e.g. paged decode attention
   (custom Metal via `torch.mps` or an MLX backend), only after 1 and 2 are
   measured.

## Throughput

4. **Correct batched serving.** Batched prefill samples every sequence;
   per-sequence sampling params in the engine sampler; MPS/CPU attention
   reads prefix-cached blocks. Then let the server batch concurrent requests
   and keep the prefix cache across requests.
5. **Continuous batching with mixed prefill/decode steps** and chunked
   prefill, so a long prompt doesn't stall running decodes.

## Model coverage

6. **Model dispatch on `model_type`** and a second architecture (e.g. Llama
   or Qwen2.5), each with parity.
7. **Larger Qwen3 sizes** (1.7B, 4B, 8B) on MPS within memory, then
   quantized weights (int8 / int4) to reach the sizes mlx-lm runs.

## Infra

8. A CUDA runner for parity and benchmarks on the flash-attn path.
9. `black --check` in CI once the tree is formatted in one commit.
