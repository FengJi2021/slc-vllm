# Model execution

Runs one scheduled batch through the model and samples the next tokens.
Source: [`dvllm/engine/run_model.py`](../../dvllm/engine/run_model.py),
[`dvllm/models/`](../../dvllm/models/), [`dvllm/layers/`](../../dvllm/layers/),
[`dvllm/utils/`](../../dvllm/utils/).

## RunModel

`RunModel.__init__` picks the device, sets the default dtype to the HF
config's `torch_dtype` (bf16 for Qwen3), builds the model, loads weights
(`utils/loader.py`), warms up (`warmup_model`, smaller on MPS), allocates
the KV cache, and on CUDA without `enforce_eager` captures CUDA graphs.

`run(seqs, is_prefill)`:

1. `prepare_prefill` / `prepare_decode` flatten the batch into `input_ids`,
   `positions`, `cu_seqlens_*`, `slot_mapping` (where each token's K/V goes
   in the cache), `context_lens` and `block_tables`, and store them in the
   global context (`utils/context.py`) that attention layers read.
2. `run_model` runs the model and computes logits (last position per
   sequence).
3. The sampler picks tokens (rank 0 only).

With `tensor_parallel_size > 1`, rank 0 sends each call to the other ranks
over shared memory (`write_shm` / `loop`).

## Model

`Qwen3ForCausalLM_V2` in [qwen3_v2.py](../../dvllm/models/qwen3_v2.py)
(`TransfomerBlock` → `Qwen3Model` → `Qwen3ForCausalLM_V2`). Long prefills on
MPS/CPU go through `_chunked_attention`, which masks each query chunk
against the full key history.

## Attention backends

| Device | Prefill | Decode | KV write |
| --- | --- | --- | --- |
| CUDA + flash-attn | `flash_attn_varlen_func` | `flash_attn_with_kvcache` | Triton `store_kvcache` |
| MPS / CPU | `F.scaled_dot_product_attention`, chunked for long prompts | SDPA over the gathered cache | Python loop per token |

Both live in [layers/attention.py](../../dvllm/layers/attention.py) and the
model's `_sdpa` / `_chunked_attention`. A change to one row is checked
against the other. The per-token Python KV write is the main MPS prefill
cost today.

## Sampling

[layers/sampler.py](../../dvllm/layers/sampler.py): temperature +
Gumbel-max, one temperature for the whole batch, and greedy isn't allowed
(`SParams` asserts `temperature > 1e-10`). The server replaces it with
`RequestSampler` (greedy, top-p, top-k, penalties) for one sequence.

## Debugging numerics

When output is nonsense or NaN:

1. Run `test/parity.py` to confirm the model diverges from the reference,
   and on which prompt.
2. Turn on per-layer checks for a range of layers:
   `DV_DEBUG_LAYERS="0-5" uv run python test/parity.py --device mps 2>&1 | grep -E "WARNING|NaN|Inf"`.
   `_debug_sdpa` computes mean/std/min/max of `q`, `k`, `v` and the
   attention scores (`sc`) and logs `[WARNING] <name> attention range is
   abnormal` when one falls outside `_attention_range_checker`'s bounds
   (uncomment its `logger.debug` line for the raw numbers). Move the range
   deeper until you find the first bad layer.
3. Exploding scores (`sc` max past ~88 overflows `exp` in fp16) mean
   overflow or a wrong scale. Rerun in float32
   (`torch.set_default_dtype(torch.float32)` in `RunModel.__init__`): if
   fp32 is fine, the bug is precision or the scale (`1/sqrt(head_dim)`),
   not the logic. Large `q`/`k` after the norms point at `RMSNorm` or the
   weights.
4. `tools/get_params.py` lists the checkpoint's weight names and shapes, to
   check them against the model's parameters.

## Adding a model

1. Add `dvllm/models/<name>.py` with a `...ForCausalLM` exposing
   `forward(input_ids, positions)` and `compute_logits`, built from
   `dvllm/layers/` so it uses the paged KV cache through the context.
2. Map its weights in `utils/loader.py` (`load_model` merges HF's separate
   `q_proj`/`k_proj`/`v_proj` into `qkv_proj` by name; a model with other
   names or fused layers needs its own mapping).
3. Select it in `RunModel.__init__`. Today `Qwen3ForCausalLM_V2` is
   hard-coded there; the first new model adds a dispatch on
   `hf_config.model_type`.
4. Make `test/parity.py` pass against HF transformers with the model's chat
   template, on CPU and MPS, and record its numbers in
   [benchmarks.md](../benchmarks.md).
