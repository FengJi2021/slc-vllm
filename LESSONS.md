# Lessons from merged work

What earlier work got wrong, and the rule that would have caught it. Read
this before changing code. Each lesson is a rule, then why, then the
evidence (commit, file or issue). "Seen" counts the separate times it
happened; a lesson seen 3 times moves into
[docs/code-standards.md](docs/code-standards.md).

Add a lesson when a fix had to be fixed again, when a bug reached `main`, or
when a check would have caught a problem sooner. Keep each one short.

## Numerics

**Check a new attention path against a reference before reading anything
else into its output.**
- The chunked MPS attention ignored the KV history of earlier chunks, and the
  model produced nonsense until it was found layer by layer
  (`dvllm/models/qwen3_v2.py` `_chunked_attention`).
- `test/parity.py` now compares greedy tokens with HF transformers; run it
  first.
- Seen 1×.

**Exploding activations in fp16/bf16 are tested in fp32 first.**
- Rerun in float32 (`torch.set_default_dtype(torch.float32)` in
  `RunModel.__init__`). If fp32 is fine, the bug is overflow or the scale,
  not the logic. Use `DV_DEBUG_LAYERS` to find the first bad layer
  ([Debugging numerics](docs/subsystems/model-execution.md#debugging-numerics)).
- Seen 1×.

## Batching

**A path that runs one sequence correctly isn't correct for a batch until a
batch is tested.**
- Multi-sequence prefill samples only the last position of the flattened
  batch; the built-in `Sampler` applies one temperature to the whole batch;
  the non-CUDA prefill attention ignores prefix-cached blocks
  (`dvllm/server/worker.py` docstring). The server serializes requests to
  stay correct.
- Test batch sizes 1 and >1, with different prompt lengths and sampling
  settings per sequence, and a shared prefix.
- Seen 1×.

## Memory

**Size warmup and the KV cache for the device that runs it.**
- Warmup with the CUDA-sized batch ran out of memory on MPS (964ac08); MPS
  now warms up with one sequence of at most 256 tokens
  (`RunModel.warmup_model`).
- Seen 1×.

## Code paths nobody runs

**A path CI never runs is assumed broken.**
- `LLMEngine` started tensor-parallel workers with `target=ModelRunner`, a
  name that didn't exist, so `tensor_parallel_size > 1` crashed. The fatal
  lint in CI (`flake8 --select=E9,F63,F7,F82`) catches undefined names now.
- Seen 1×.
