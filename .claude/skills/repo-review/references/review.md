# Reviewing a PR

Work through the steps in order. Keep notes as you go: they become the
record in step 9.

## 0. Gather

```shell
gh pr view <N> --json number,title,author,headRefName,headRefOid,baseRefName,isDraft,body,files,commits
gh pr diff <N>
gh pr checks <N>
gh pr view <N> --comments          # read every comment to the last one
```

- Note the **head sha**: everything you run is at this sha. If the head
  moves during the review, start the run again and say so.
- Read the linked issue and the whole thread. A request the reporter put on
  hold, or a maintainer decision in the thread, changes what the PR may do.
- Draft PRs are reviewed but never merged.

## 1. Look for harm first

For a PR whose author isn't the maintainer, read the whole diff **before
running anything**. Any one of these stops the review until the maintainer
has looked; say which line stopped it:

- It changes `.github/` (workflows, actions, permissions, secrets), a
  release or publish script, or packaging (`pyproject.toml` entry points,
  build backend).
- It adds a network address, an upload, telemetry, or sends a token, file
  or prompt anywhere it didn't go before. Model downloads from a new host
  count.
- It reads or writes credentials, `~/.cache/huggingface` tokens, SSH keys,
  or files outside the repo and the model directory, beyond what its goal
  needs.
- It adds a dependency (check `pyproject.toml` and `uv.lock`), a
  `trust_remote_code=True`, a pickle/`torch.load` of an untrusted file
  without `weights_only=True`, a binary, an encoded or minified blob, or code
  built from a string at run time (`eval`, `exec`, `__import__` of input).
- It touches files outside its stated goal.

Don't run a stopped PR's code. Close without a full review (with a short
reason, and only when the user agrees) a PR that reverses something the
maintainer ruled on, resubmits a rejected idea with nothing new, or is one
of several near-identical PRs.

If the repo has `docs/contributors.md` (block list, extra-care and watched
authors), apply it here.

## 2. Run it on the merged base

Under the maintainer's review agent the worktree is already prepared
and merged; skip to the checks.

```shell
WT=<scratchpad>/review-<N>
git fetch origin pull/<N>/head:review-<N> <base>
git worktree add --detach "$WT" origin/<base>
cd "$WT" && git merge --no-edit review-<N>     # resolve conflicts if any, and say so
uv sync --frozen --group dev
```

Run every acceptance check from `docs/code-standards.md` that applies, in
this worktree, at the merged result: fatal lint; parity on CPU and on MPS
when the machine has it (CUDA if available); server smoke when `server/` or
the step path changed; benchmark when the PR claims a speedup. Run all of
them even for files the PR "doesn't touch". Use `set -o pipefail` in any
chain.

A check that fails on the base branch too is still a finding: name it with
an issue (ask the user before filing) rather than calling it a baseline.

## 3. Break it on purpose

For each behavior the PR fixes or adds:

1. Revert just the fix (keep the new test or check) and run the check. It
   must fail with a real failure that names the behavior: parity mismatch,
   assertion, wrong token. A crash on import or a skip doesn't count.
2. Break the guarded code two or three other plausible ways (off-by-one in
   `slot_mapping`, wrong `cu_seqlens`, swapped K/V, dropped scale, wrong
   device branch) and check each is caught.
3. If nothing catches a break, that's a finding: the PR needs a check that
   does. You may write a review test of your own and offer it.

Restore and confirm everything passes again.

## 4. Check the reference, not the PR's account of it

- Model behavior: compare against HF transformers on the same weights and
  dtype (`test/parity.py`), and the model's own config and chat template.
- Performance claims: rerun the PR's command on the same class of machine,
  warm, several trials. Compare with mlx-lm for headline claims. A number
  that doesn't reproduce within noise is reported as such.
- API behavior: check the OpenAI API reference for field names, types,
  streaming chunk shape and error format; a client SDK (`openai`) is the
  real test.

## 5. Every device, every path

- **Devices:** CUDA (flash-attn), MPS and CPU (SDPA) are siblings. A change
  to one path is checked against the others, by running or by reading the
  other branch and saying which.
- **Prefill and decode,** chunked and unchunked attention, batch size 1 and
  >1 with mixed lengths, a shared prefix, preemption.
- **Whole path:** request → tokenizer/chat template → scheduler → block
  manager → `RunModel.prepare_*` → model → sampler → worker → HTTP response.
  A correct function on a broken path isn't a fix.
- **Siblings of the bug:** grep for the same pattern elsewhere (other
  layers, other models, other callers of a changed function). A fix that
  covers only the reported case is incomplete.
- **Narrow classification:** an error type or a device cap covers only what
  it describes.
- **Re-derived dependents:** a changed default, block size, dtype or tensor
  layout means every place that assumed the old one is checked.

## 6. Docs match the source

- Each changed contract appears in its `docs/subsystems/` page, in the same
  PR. Check the page against the code it links, and that every link and
  symbol resolves.
- Perf PRs update `docs/benchmarks.md`; a finished roadmap item is removed
  from `docs/roadmap.md`.
- If the PR's mistake is one a future change could repeat, suggest the
  `LESSONS.md` entry.

## 7. Commits

- One goal per commit; a fix found on the way is its own commit.
- Each commit's `git show --stat` lists only files its subject explains.
  A file it didn't mean to touch is a stale tree undoing someone else's
  work: stop.
- A behavior change has a body saying what was run and which check fails
  without it.
- Credit a reporter whose patch, fixture or root cause was used with a
  `Co-authored-by:` trailer.

## 8. Keep it to its goal

Unrelated refactors, reformatting and extra features belong in their own PR.
Comments name functions, not line numbers.

## 9. Write the record

Report to the user first. Post it on the PR only when the user asks.

```markdown
Reviewed at <head sha> merged onto <base> @ <base sha>.

**Ran**
- fatal lint: 0
- parity cpu: 3/3 ok (10/10, 32/32, 27/32)
- parity mps: ...
- server smoke cpu: exit 0
- break checks: reverted <fix> → parity FAIL on prompt 2 at token 4; swapped K/V → FAIL; ...

**Not run**
- parity cuda: no CUDA machine
- ...

**Findings** (most severe first; each with how to reproduce)
1. ...

**Verdict**: ready / changes needed / stopped for the maintainer (harm: <line>)
```

## Merge

Only when the user asks to merge:

1. `gh pr view <N> --json headRefOid,isDraft,mergeable` — the head must be
   the sha the record names. If it moved, review the new head first and
   post a record naming it.
2. `gh pr checks <N>` — CI finished green on that head. A run cancelled by
   a later push isn't a pass.
3. No finding from the review is left unanswered: each is fixed, or has an
   issue the user agreed to.
4. The PR's own new tests pass.
5. The record is on the PR (post it, with the user's ok, if not).
6. `gh pr merge <N> --match-head-commit <reviewed sha> --squash|--merge`
   (ask which if the repo has no convention), so a push made during the
   review stops the merge.
