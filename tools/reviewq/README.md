# reviewq: approval-gated local reviews

How a PR is reviewed here:

1. **First pass (GitHub, automatic).** `.github/workflows/first-pass.yml`
   reads each new PR (it doesn't run anything) and posts one comment: harm
   scan, findings, docs vs source, and whether a local review is worth the
   machine time. CI runs lint, parity and the server smoke test on CPU.
2. **Approval (you).** Only PRs you approve get a local review: add the
   `local-review` label on GitHub, or run `reviewq approve <PR#>`. Approval
   covers one head commit; after a new push, approve again.
3. **Local review (this Mac, queued).** A `tick` every 15 minutes queues
   approved PRs and starts one review at a time when there's capacity. The
   review is Claude Code running `/repo-review review <PR#>` unattended in a
   fresh worktree of the PR merged onto its base. It runs the checks with
   MPS, breaks the fix on purpose, and writes a record. It posts nothing.
4. **Decision (you).** Read the report, then post it, ask follow-ups, or
   merge yourself.

## Setup

```shell
python3 tools/reviewq/reviewq.py setup      # ~/.slc-review: config, memory, a clone; GitHub labels
python3 tools/reviewq/reviewq.py install    # launchd agent: tick every tick_minutes
```

For the first pass, run `claude setup-token` and save the token as the
repository secret `CLAUDE_CODE_OAUTH_TOKEN`
(`gh secret set CLAUDE_CODE_OAUTH_TOKEN`). Without it the workflow skips.

An alias helps: `alias reviewq='python3 ~/github/slc-vllm/tools/reviewq/reviewq.py'`.

## Daily use

| Command | What it does |
| --- | --- |
| `reviewq approve 12` | Approve PR #12 for a local review (adds the label, queues it) |
| `reviewq queue` | Pending, running and recent jobs, and why the next one is waiting |
| `reviewq run 12` | Start PR #12's queued review now, ignoring the window, cap and load |
| `reviewq cancel 12` | Withdraw the approval |
| `reviewq report 12` | Print the latest report and how to resume the session for follow-ups |
| `reviewq post 12` | Show the report and, if you say yes, post it on the PR |
| `reviewq lessons` | Queue a lessons review (works on branch `lessons/<date>` in the clone; nothing is pushed) |
| `reviewq memory proposals` | What reviews proposed to remember |
| `reviewq memory accept job7 [--edit]` / `reject job7` | Your decision on a proposal |
| `reviewq memory show` | The reviewer's memory |

## Capacity

`~/.slc-review/config.json`:

| Key | Default | Meaning |
| --- | --- | --- |
| `window` | `00:00-24:00` | Local hours when a run may start, e.g. `22:00-07:00` |
| `max_runs_per_day` | `3` | Runs started per day |
| `max_load_per_cpu` | `0.5` | Wait while the machine is busier than this |
| `require_ac_power` | `true` | Wait while on battery |
| `timeout_minutes` | `90` | A run past this is stopped and marked failed |
| `tick_minutes` | `15` | launchd interval (`reviewq install` again after changing it) |
| `lessons_every_days` | `0` | Queue a lessons review every N days; 0 means only by hand |
| `model` | unset | `claude --model` for reviews |

One review runs at a time (`run.lock`). Reviews use your Claude login: on a
subscription they draw on the plan's usage, so the daily cap also protects
your limits.

## Memory

`~/.slc-review/memory/MEMORY.md` goes into every review's prompt. The
reviewer can't write it: its settings deny writes to the memory folder, and
it can only end its report with a `## Proposed memory` section. reviewq saves
that section to `memory/proposals/`, and nothing reaches memory until you
accept it. You can also edit `MEMORY.md` by hand.

Memory is for what helps the next review on this machine (check timings,
flaky cases, your preferences). Rules for everyone go in `LESSONS.md`
through a PR.

## What the reviewer may do

`reviewer-settings.json` runs Claude in `dontAsk` mode: only listed tools
run, everything else is refused without a prompt. It may read, run git
read commands, `gh pr view/diff/checks`, `uv sync/run`, the server and
checks, and edit only inside `~/.slc-review/work/`. It may not push, merge,
comment, label, call `gh api`, or touch its memory, the queue or
`~/.claude`.

These rules prevent accidents. They aren't a sandbox: the review runs the
PR's code (`uv run`) with your user's rights. That's why a local review
needs your approval, and why the first pass flags changes to the files a
review runs with (`.claude/`, `AGENTS.md`, `tools/reviewq/`, ...). Read
what the first pass flags before approving.

## Files

`~/.slc-review/`: `config.json`, `queue.json`, `run.lock`, `repo/` (a
separate clone; your checkout is never touched), `work/pr<N>` and
`work/lessons` (worktrees), `reports/`, `memory/`, `logs/` (`reviewq.log`,
`job<N>.log`, `launchd.log`).
