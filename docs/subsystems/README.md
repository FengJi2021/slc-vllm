# Subsystem references

What each part is responsible for, its runtime path, its state and its
contracts. Start with the page for the part you change, then follow its
source links to check the implementation.

| Subsystem | Reference |
| --- | --- |
| Request lifecycle: `LLMEngine`, `Scheduler`, `BlockManager` (paged KV cache, prefix hashing), `Sequence` | [Engine](engine.md) |
| `RunModel`, the model (`Qwen3ForCausalLM_V2`), attention backends per device, sampler, weight loading, adding a model | [Model execution](model-execution.md) |
| `slc-server`: the OpenAI-compatible API, the single-sequence worker, streaming, stop strings, reasoning and tool calls | [Server](server.md) |

## Maintain a reference

Update the owning page in the same PR when a change alters a documented
responsibility, interface, state transition, runtime path or failure
behavior. A refactor that keeps these contracts doesn't need a prose change.

Keep each fact in one page and link to it from elsewhere. Link to source
files and named symbols rather than copying code that changes often.
Document current behavior; proposals belong in the PR or
[roadmap](../roadmap.md).

## Describe a PR's semantic change

For each affected subsystem, say what triggers the change and the behavior
before and after, with the evidence (parity output, smoke output, benchmark
numbers). Say so when a refactor keeps behavior the same.

## Review a change

1. Find the affected subsystem and the real runtime path for each device
   (CUDA flash-attn vs MPS/CPU SDPA, prefill vs decode).
2. Compare the claimed before/after with the base and head code, beyond the
   changed lines: callers, state and the other device's path.
3. Check that changed contracts appear in the owning page.
4. Check that the cited evidence exercises the claim. Report checks not run.
