# Code standards

What a reviewer, human or agent, checks on every change. How to describe and
review a change to one subsystem is in the
[subsystem references](subsystems/README.md).

## What a change contains

| Rule | What a reviewer looks at |
| --- | --- |
| A change does only what its goal needs. | Unrelated refactors, reformatting and extra features go in a PR of their own. |
| A change holds on the whole path. | Follow it from the request to the sampled token: tokenizer, scheduler, block manager, `RunModel.prepare_*`, model, sampler, server. One correct function doesn't make the step correct. |
| A fix reaches every sibling. | CUDA and MPS/CPU paths, prefill and decode, chunked and unchunked attention, and every caller of a changed function get the same fix in the same change. |
| A change re-derives what is built on it. | Changing a block size, a tensor layout or a config default means finding every place that assumed the old value (`config.py`, `run_model.py`, `server/cli.py`, tests). |
| Device-specific limits stay device-specific. | An MPS memory cap or a CUDA-only kernel is guarded by the device, and the other devices are checked unchanged. |

## Correctness

| Rule | What a reviewer looks at |
| --- | --- |
| Greedy output matches the reference. | `test/parity.py` passes on every device the change touches. A lowered `--min-match` needs a reason. |
| A fix comes with a check that fails without it. | Put the old code back, run the check, and say what it failed with. |
| Batch paths are tested as batches. | Batch size >1, mixed prompt lengths, per-sequence sampling settings, shared prefixes. See [LESSONS.md](../LESSONS.md#batching). |
| A new model passes parity before it's merged. | Against HF transformers on the same weights and dtype, with its chat template. |

## Performance

| Rule | What a reviewer looks at |
| --- | --- |
| A perf claim has before/after numbers. | Same machine, model, command and dtype. Name the chip and memory, the commit, and the command. Prefill tok/s, decode tok/s, and TTFT where it applies. |
| A speedup on one metric doesn't cost another silently. | Report TTFT, decode tok/s and peak memory, and say if any got worse and why that's acceptable. |
| Compare with mlx-lm where it helps. | For a headline change on Apple Silicon, run mlx-lm on the same model and record both in [benchmarks.md](benchmarks.md). |
| Warm runs only. | Exclude model load and the first (warmup) iteration; report the average of several trials. |

## Acceptance

A PR says which checks ran and what they printed. A check that wasn't run is
listed as not run, never left out.

1. Fatal lint passes.
2. `test/parity.py` passes (CI runs CPU; run MPS locally when MPS code changed).
3. Server smoke passes when the server or the step path changed.
4. Benchmark numbers, when the PR claims a speedup.

## Commits

- One logical change per commit, with a subject that says what changed.
- A commit that changes behavior says in its body what was run to verify it.
