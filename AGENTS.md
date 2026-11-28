# Notes for coding agents

## What this repo is for

`dvllm` is a small, readable LLM inference engine in PyTorch (paged KV
cache, prefix-hashing block manager, prefill/decode scheduler, hand-written
models) with an OpenAI-compatible server (`slc-server`). The primary target
is Apple Silicon (MPS); CUDA and CPU also run.

Every change should move one of three numbers, and none of them backwards:

1. **Model coverage:** more architectures and sizes run correctly.
2. **Throughput:** more tokens/s across concurrent requests.
3. **Latency:** lower TTFT and lower time per output token.

The yardstick on a Mac is the official [mlx-lm](https://github.com/ml-explore/mlx-lm)
on the same model and machine. [docs/benchmarks.md](docs/benchmarks.md)
records where we stand, and [docs/roadmap.md](docs/roadmap.md) lists the
known gaps in priority order.

## Before changing code

- Read [LESSONS.md](LESSONS.md): what earlier work got wrong and the rule
  that would have caught it.
- Read the [subsystem reference](docs/subsystems/README.md) for the part you
  touch, then verify it against the linked source.
- Write and review changes against the [code standards](docs/code-standards.md).

## Verify

The model lives at `~/huggingface/Qwen3-0.6B` (`uv run python download_model.py`).

| Check | Command | When |
| --- | --- | --- |
| Fatal lint | `uv run flake8 --select=E9,F63,F7,F82 dvllm test tools` | every change |
| Greedy parity vs HF | `uv run python test/parity.py --device mps` (and `--device cpu`) | every change to `engine/`, `layers/`, `models/` |
| Server smoke | `uv run slc-server --model ~/huggingface/Qwen3-0.6B` then `uv run python test/server_smoke.py` | every change to `server/` or the engine's step path |
| Throughput | `uv run python test/benchmark.py --device mps` | every change that claims a speedup |

CI runs the lint, parity and smoke checks on CPU (GitHub's macOS runners have
no Metal), so **MPS results and every performance number come from a local
run** and go in the PR description.

## Rules that matter most here

- **Correctness before speed.** A speedup that changes greedy output is a bug.
  `test/parity.py` must pass on the device the change targets.
- **A perf claim carries numbers.** Before and after, same machine, same
  command, same model, with the device and commit. Update
  [docs/benchmarks.md](docs/benchmarks.md) when a change moves a headline number.
- **Every device path is a sibling.** `layers/attention.py` and
  `models/qwen3_v2.py` have a CUDA (flash-attn) path and an MPS/CPU (SDPA)
  path. A fix to one is checked against the other.
- **The server serializes on purpose.** `dvllm/server/worker.py` explains why
  batched serving isn't correct yet. Don't enable batching in the server
  until the batched engine path passes parity for every sequence in the batch.
- **Keep it readable.** This is also a learning codebase: prefer a clear
  implementation with a comment on *why* over a clever one.

## Reviews

Every PR gets a read-only first pass on GitHub. A full local review runs
only when the maintainer approves it: see [tools/reviewq](tools/reviewq/README.md).

## After a change

If you learned something a future change would otherwise get wrong, add it to
[LESSONS.md](LESSONS.md) in the same PR. If you changed a documented
contract, update its subsystem reference in the same PR.
