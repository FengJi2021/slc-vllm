## What and why

<!-- The goal, and for each affected subsystem (docs/subsystems/) what triggers
the change and the behavior before and after. Say so if behavior is unchanged. -->

## Verification

<!-- What ran and what it printed. List checks that were not run as "not run". -->

- [ ] Fatal lint
- [ ] `test/parity.py` on: cpu / mps / cuda
- [ ] Server smoke (if `server/` or the step path changed)

## Performance

<!-- Required if this claims a speedup. Same machine, model, command, dtype.
Delete if not applicable. -->

| | Before | After |
| --- | --- | --- |
| Machine / device | | |
| TTFT (s) | | |
| Prefill tok/s | | |
| Decode tok/s | | |
| Peak memory | | |

Command:

## Docs

- [ ] Subsystem reference updated (or no contract changed)
- [ ] `LESSONS.md` entry added (if something here was learned the hard way)
- [ ] `docs/benchmarks.md` updated (if a headline number moved)
