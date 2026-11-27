# Engine

Turns prompts into token sequences, one `step()` at a time. Source:
[`dvllm/engine/`](../../dvllm/engine/), config in
[`dvllm/config.py`](../../dvllm/config.py).

## Parts

- **`LLMEngine`** ([llm_engine.py](../../dvllm/engine/llm_engine.py)) builds
  the `Config`, the tokenizer, the `RunModel` (rank 0, plus spawned ranks for
  `tensor_parallel_size > 1`) and the `Scheduler`. `step()` asks the
  scheduler for a batch, runs it, and hands the sampled tokens back.
  `generate()` loops `step()` until every sequence finishes. `LLM` in
  [llm.py](../../dvllm/llm.py) is an alias.
- **`Sequence`** ([seq.py](../../dvllm/engine/seq.py)) holds the prompt and
  completion token ids, status (`WAITING`, `RUNNING`, `FINISHED`), its block
  table, and `num_cached_tokens` (prefix-cache hits).
- **`Scheduler`** ([scheduler.py](../../dvllm/engine/scheduler.py)) keeps
  `waiting` and `running` deques. Each `schedule()` returns either a
  **prefill** batch (new sequences from `waiting`, bounded by
  `max_num_seqs`, `max_num_batched_tokens` and free blocks) or, if there is
  none, a **decode** batch (one token for each running sequence). Prefill
  and decode are never mixed in one step. When a decode can't get a block,
  it preempts the newest running sequence back to the front of `waiting`
  (its blocks are freed; it is prefilled again later).
- **`BlockManager`** ([block_manager.py](../../dvllm/engine/block_manager.py))
  owns the paged KV cache's block ids (`kvcache_block_size`, 256 tokens,
  must be a multiple of 256). A full block is keyed by an xxhash of its
  tokens chained with the previous block's hash, so sequences sharing a
  prefix share blocks (`ref_count`). `allocate` on prefill, `may_append` on
  decode (a new block when the sequence crosses a block boundary; hashing
  the block when it fills), `deallocate` on finish or preempt.

## KV cache sizing

`RunModel.allocate_kv_cache` sets `num_kvcache_blocks`: from free memory and
`gpu_memory_utilization` on CUDA; the configured value if one is given
(the server's `--kv-blocks`); else at most 4 blocks on MPS and 64 on CPU.
`Config.__post_init__` also caps MPS at `max_num_seqs=16`,
`max_num_batched_tokens=2048` and `max_model_len=2048`.

## Known limits

- Prefix-cache hits are correct only with the CUDA attention path; the
  MPS/CPU prefill attention recomputes from the prompt and ignores cached
  blocks. The server clears `hash_to_block_id` before each request.
- Batched prefill samples one token for the whole batch
  (see [Model execution](model-execution.md#sampling)).

## Verification

`test/parity.py` (greedy tokens vs HF transformers, one sequence at a time),
`test/benchmark.py` (prefill and decode tok/s over batch sizes).
