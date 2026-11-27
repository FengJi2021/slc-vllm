# The lessons review

Each day's (or week's) commits are reviewed for what went wrong, and what
was learned is kept in `LESSONS.md`. A lesson seen on 3 separate days becomes a rule in
`docs/code-standards.md`. This is how the repo gets better at not repeating
itself. Run it daily when the repo is busy, or every week or so when it's
quiet. It can be scheduled (`/schedule`) once the user wants that.

## 1. Find the range

`LESSONS.md` starts with a marker naming the last commit reviewed:

```
<!-- reviewed-through: <sha> (<YYYY-MM-DD HH:MM +zone>) -->
```

If it's missing, ask the user where to start (default: the last 2 weeks),
and add it at the end. With an argument, review from that sha or date.

```shell
git fetch origin
git log --reverse --format='%h %ad %an %s' --date=iso <marker>..origin/<default branch>
gh pr list --state merged --search "merged:>=<marker date>" --json number,title,mergedAt,headRefOid
gh run list --branch <default branch> --limit 50 --json headSha,conclusion,status,createdAt
```

## 2. Look for three things

**Fixed again.** A commit that fixes, reverts or follows up an earlier
commit in the range or just before it.
- Subjects with fix/revert/follow-up/again/regression, `git log -S` on the
  lines a fix changed, and commits touching the same function within days.
- For each: what did the first commit get wrong, and what check (run,
  reference, device, sibling) would have caught it before merge?

**Shouldn't have merged.** A merge or push to the default branch with its
own new test failing, CI red or cancelled at that head, a review finding
unanswered, a record naming a different head, or a parity or benchmark claim
that doesn't reproduce.

**Verified too thinly.** A behavior change with no body saying what was run;
a perf claim without before/after numbers; MPS code changed with only CPU
checked; a test that passes with the fix reverted (try it on a sample); a
check that wasn't run and wasn't listed as not run.

Reproduce what you can. Don't record a suspicion as a lesson; record it as
a question for the user.

## 3. Write the lessons

Each lesson has this shape, under the section it fits (add one if needed):

```markdown
**<The rule, as an instruction, bold.>**
- <commit/PR>: what happened, and the commit that fixed it.
- <what to do instead, concretely: the command, the device, the check>.
- Seen 1× (<YYYY-MM-DD>).
```

- Same mistake as an existing lesson, on a **different day**: add the
  evidence and bump `Seen N× (date, date, ...)`. Same day doesn't count again.
- A lesson that was done right in this range can say so ("Done right on
  <date>: <commit>"): it shows what good looks like.
- At the top, under the intro, add one short paragraph for the range:
  commits, merges, what mostly went right, what went wrong.
- Keep lessons short. Evidence is commit shas and PR numbers, not prose.

## 4. Promote

When a lesson reaches **Seen 3×** on separate days:

1. Add it as a row to the right table in `docs/code-standards.md`
   (`| Rule | What a reviewer looks at |`), with its evidence trimmed to
   the strongest examples.
2. In `LESSONS.md`, replace the lesson's body with one line pointing to the
   standard, and list it under `## Moved to the code standards` with the
   three dates.
3. If a check can enforce it (a CI step, a lint rule, a parity case, a test),
   propose that too: a rule a machine checks is better than one people
   remember.

## 5. Finish

- Move the `reviewed-through` marker to the last commit reviewed.
- Put everything in one commit: `lessons: review <from>..<to>`, with a
  body listing the new lessons, the bumps and any promotion.
- Show the user the diff. Push or open a PR only when they ask.
- List open questions (suspicions you couldn't reproduce) separately.
