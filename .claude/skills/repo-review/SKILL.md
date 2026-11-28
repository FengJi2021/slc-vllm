---
name: repo-review
description: Run this repo's full review process - review a PR by running it and breaking it on purpose, record and merge only the reviewed head, run the periodic lessons review that grows LESSONS.md and promotes rules into the code standards, screen outside contributions for harm, or grow the repo's agent docs from the minimal set to the full layout. Use when the user types /repo-review with a mode (review, merge, lessons, harm, adopt).
disable-model-invocation: true
argument-hint: "review <PR#> | merge <PR#> | lessons [since] | harm <PR#> | adopt"
---

# repo-review

How this repo lets agents write, review and merge code without the quality
drifting. The repo is a PyTorch inference engine whose goal is more models,
more throughput and lower latency, measured against mlx-lm.

The idea in one paragraph: **every review runs the code, every check that
wasn't run is said out loud, every mistake that reaches `main` becomes a
written rule, and a rule that keeps being broken becomes a standard.** The
repo carries its own memory: `AGENTS.md` routes, `LESSONS.md` remembers,
`docs/code-standards.md` enforces, `docs/subsystems/` describes, CI guards.

## Before any mode

Read, in this order, and treat them as the source of truth over this skill
where they are more specific:

1. `AGENTS.md`: goals, check commands, top rules.
2. `LESSONS.md`: rules learned from earlier mistakes.
3. `docs/code-standards.md`: what every change is held to.
4. `docs/subsystems/README.md` and the page for each subsystem touched.

Then pick the mode from the arguments. With no arguments, ask which mode.

| Mode | What it does | Reference |
| --- | --- | --- |
| `review <PR#>` | Full review: harm scan (outside authors), run it on the merged base, break it on purpose, follow the whole path on every device, check docs against source, write a record. Never merges. | [references/review.md](references/review.md) |
| `merge <PR#>` | Only when the user asked to merge: verify the record names the current head and CI is green, then `gh pr merge --match-head-commit`. | [references/review.md#merge](references/review.md#merge) |
| `lessons [since]` | Review commits since the `reviewed-through` marker: fixes of fixes, things that shouldn't have merged, thin verification. Write or bump lessons, promote 3-day lessons into the code standards. | [references/lessons-review.md](references/lessons-review.md) |
| `harm <PR#>` | Only the "look for harm first" scan, for a PR from someone other than the maintainer. | [references/review.md#1-look-for-harm-first](references/review.md#1-look-for-harm-first) |
| `adopt` | Bring the repo from the minimal setup to the full layout, one piece at a time, asking before each. | [references/repo-model.md](references/repo-model.md) |

## Rules for every mode

- **Run it; reading the diff isn't a review.** A finding you didn't
  reproduce is reported as unverified.
- **Say what wasn't run.** Every record lists checks run (with what they
  printed, at which commit) and checks not run (with why). Never leave one
  out.
- **The record is written once the run has ended.** No "still running".
- **A red check on the base branch is a bug now,** not a baseline. Name it
  or file it; don't ship past it.
- **Don't reverse the maintainer's decisions.** A change to something the
  maintainer ruled on (e.g. the server serializes requests on purpose) is
  left for the maintainer.
- **Outward actions need the user.** Posting a review comment, merging,
  closing, labelling or filing an issue happens only when the user asked
  for it in this session. Never merge unprompted.
- **Work in your own checkout.** Use `git worktree add --detach` under the
  scratchpad for a PR head (the local queue prepares one for you); don't
  switch the user's working branch or touch their uncommitted files.

## Where it runs

Locally, on the maintainer's machine. A full `review` runs only for PRs
whose CI is green and that the maintainer approves, queued by the
maintainer's local review agent, or when the maintainer types the command.
Step 1 (look for harm) always comes before running any of the PR's code.

## Repo-specific checks (from AGENTS.md)

| Check | Command |
| --- | --- |
| Fatal lint | `uvx flake8 --count --select=E9,F63,F7,F82 dvllm test tools` |
| Greedy parity | `uv run python test/parity.py --device {cpu,mps,cuda}` |
| Server smoke | start `uv run slc-server --model ~/huggingface/Qwen3-0.6B --port <free>`, then `uv run python test/server_smoke.py --base-url http://127.0.0.1:<port>/v1` |
| Throughput | `uv run python test/benchmark.py --device mps` (before and after, same machine) |
| mlx-lm reference | `uvx --from mlx-lm mlx_lm.generate --model Qwen/Qwen3-0.6B --prompt ... --max-tokens 160` |

CI has no Metal: MPS and all performance numbers come from a local run.
If `AGENTS.md` lists different commands, use those.
