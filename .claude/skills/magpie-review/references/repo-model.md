# The repo model, and adopting the rest of it

## How magpie's repo is laid out

| File | Role | How it grows |
| --- | --- | --- |
| `CLAUDE.md` | `@AGENTS.md` + `@LESSONS.md`, nothing else, so every agent tool reads the same thing | Never; edit what it imports |
| `AGENTS.md` | A short router: what to read before which kind of work, plus facts too important to be one link away | When a new kind of work needs its own pointer |
| `LESSONS.md` | Rule → why → evidence from recent mistakes; `Seen N×`; a `reviewed-through` marker | The lessons review (lessons-review.md) |
| `docs/code-standards.md` | Rule tables (what a reviewer looks at), tests, acceptance criteria, reviewing a PR, commits | Lessons seen 3 days are promoted here |
| `docs/subsystems/` | One page per subsystem: responsibilities, source of truth, runtime paths, state, constraints, verification; plus rules for maintaining pages and describing/reviewing changes | Same PR as any contract change |
| `docs/contributors.md` | Harm-first screening, close-unreviewed rules, credit, block list, extra-care and watched authors with evidence | When an outside author's work needs it |
| `.github/workflows/test.yml` | CI; each step's comment says which incident put it there | A step per class of red that reached main |
| Release workflow | A release only from a green CI run on `main`; a tag isn't a release until the artifacts exist | When the repo ships versions |

The principles behind it:

- **Each fact lives in one place**, and everything else links to it.
- **A rule carries its evidence.** "Why" is a commit or PR number, so a
  later reader can check it and nobody argues from memory.
- **Mistakes become rules, rules become checks.** Lesson → standard → CI.
- **Records are part of the work.** What was run, at which commit, and what
  wasn't run, in the commit body or on the PR.
- **Agents propose, the maintainer decides** on anything irreversible or
  anything about people (blocks, product decisions, releases).

## What this repo has now (the minimal set)

`CLAUDE.md`, `AGENTS.md`, `LESSONS.md`, `docs/code-standards.md`,
`docs/subsystems/{README,engine,model-execution,server}.md`,
`docs/benchmarks.md`, `docs/roadmap.md`, `.github/workflows/test.yml`
(fatal lint; CPU parity and server smoke on macOS), and
`.github/pull_request_template.md`.

## Adopting the rest (`adopt` mode)

Check what already exists, then offer each missing piece below, one at a
time, and ask before writing each. Keep the existing files' voice: short
rules, a "what a reviewer looks at" column, and evidence.

1. **"Reviewing a PR" in `docs/code-standards.md`.** Condense
   [review.md](review.md) steps 2–8 into bullets: run it on the merged base,
   break it on purpose, check the reference, every device and path, docs
   match source, keep it to its goal. Add the merge rule
   (`--match-head-commit`, record before merge).
2. **Fuller acceptance criteria:** a check that fails without the change
   (named, with its output); all checks, not chosen ones; MPS run locally
   for MPS changes; the record written after the run ended.
3. **Commit rules:** subject says the behavior change; body has what was run;
   one goal per commit; `--stat` only lists the change's files; reporter
   credit trailers.
4. **`reviewed-through` marker** at the top of `LESSONS.md`, and the
   lessons review's intro paragraph format. Offer to schedule the review.
5. **`docs/contributors.md`** once outside contributors appear: harm-first
   list (from review.md step 1), close-unreviewed rules, credit rules,
   block/extra-care/watched lists (empty to start; an agent only proposes
   additions, with evidence).
6. **AGENTS.md pointer** for each new doc ("Before reviewing a community
   PR, read docs/contributors.md").
7. **CI hardening, as reds happen:** a comment on each step naming the
   incident; `concurrency` that never cancels `main`; step `timeout-minutes`;
   `black --check` once the tree is formatted; a CUDA runner for the
   flash-attn path; a perf-regression job on a fixed machine if one exists.
8. **Release gating** if the repo starts shipping versions: tag only a
   commit with a finished green run on `main`; check the published
   artifacts before announcing the version.
9. **Issue templates** (bug: version, device, model, steps, expected; perf
   regression: machine, command, before/after numbers).

After adopting, add a lesson only when something actually goes wrong. Don't
pre-fill rules nobody needed yet.
