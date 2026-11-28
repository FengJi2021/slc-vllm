#!/usr/bin/env python3
"""reviewq: the local, approval-gated review queue for this repo.

Only PRs the maintainer approves (the `local-review` label, or
`reviewq approve <PR#>`) are reviewed. Approved PRs wait in a queue, and a
scheduled `tick` starts one review at a time when the machine has capacity
(time window, daily cap, load, AC power). Each review runs Claude Code
unattended with the /repo-review skill in a fresh worktree of the PR merged
onto its base, writes its record to ~/.slc-review/reports, and posts nothing.
The maintainer reads the report and decides what to post.

The reviewer has a memory file that only the maintainer changes: reviews
propose additions, and `reviewq memory accept|reject` decides.

Standard library only. See tools/reviewq/README.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
STATE = Path(os.environ.get("REVIEWQ_HOME", "~/.slc-review")).expanduser()
LABEL_PLIST = "com.slc-vllm.reviewq"

DEFAULTS = {
    "repo": None,                 # owner/name; read from gh when unset
    "approve_label": "local-review",
    "done_label": "local-reviewed",
    "window": "00:00-24:00",      # local time when runs may start, e.g. "22:00-07:00"
    "max_runs_per_day": 3,
    "max_load_per_cpu": 0.5,      # 1-minute load average / CPU count
    "require_ac_power": True,
    "timeout_minutes": 90,
    "tick_minutes": 15,           # launchd interval
    "lessons_every_days": 0,      # 0: lessons reviews only when queued by hand
    "model": None,                # passed to claude --model when set
    "claude": "claude",
    "notify": True,
}


# --------------------------------------------------------------------- utils
def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def log(msg: str) -> None:
    line = f"{now()} {msg}"
    print(line)
    (STATE / "logs").mkdir(parents=True, exist_ok=True)
    with open(STATE / "logs" / "reviewq.log", "a") as f:
        f.write(line + "\n")


def sh(*args, cwd=None, check=True, capture=True) -> str:
    p = subprocess.run(args, cwd=cwd, text=True,
                       capture_output=capture, check=False)
    if check and p.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, args))} failed ({p.returncode}): "
                           f"{(p.stderr or p.stdout or '').strip()[-2000:]}")
    return (p.stdout or "").strip()


def gh_json(*args):
    return json.loads(sh("gh", *args) or "null")


def notify(cfg, title: str, msg: str) -> None:
    if cfg["notify"] and shutil.which("osascript"):
        msg = msg.replace('"', "'")[:200]
        subprocess.run(["osascript", "-e",
                        f'display notification "{msg}" with title "{title}"'],
                       capture_output=True)


# -------------------------------------------------------------------- config
def load_config() -> dict:
    cfg = dict(DEFAULTS)
    path = STATE / "config.json"
    if path.exists():
        cfg.update(json.loads(path.read_text()))
    if not cfg["repo"]:
        cfg["repo"] = sh("gh", "repo", "view", "--json", "nameWithOwner",
                         "-q", ".nameWithOwner", cwd=REPO_ROOT)
    return cfg


# --------------------------------------------------------------------- queue
def load_queue() -> dict:
    path = STATE / "queue.json"
    return json.loads(path.read_text()) if path.exists() else {"next_id": 1, "jobs": []}


def save_queue(q: dict) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = STATE / "queue.json.tmp"
    tmp.write_text(json.dumps(q, indent=2))
    tmp.replace(STATE / "queue.json")


def add_job(q: dict, kind: str, **fields) -> dict:
    job = {"id": q["next_id"], "kind": kind, "state": "pending", "added": now(), **fields}
    q["next_id"] += 1
    q["jobs"].append(job)
    return job


def jobs_for(q, pr):
    return [j for j in q["jobs"] if j.get("pr") == pr]


# ---------------------------------------------------------------------- lock
LOCK = STATE / "run.lock"


def lock_holder() -> int | None:
    try:
        pid = int(LOCK.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return None


def acquire_lock() -> bool:
    if lock_holder():
        return False
    STATE.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(str(os.getpid()))
    return True


def release_lock() -> None:
    if LOCK.exists() and LOCK.read_text().strip() == str(os.getpid()):
        LOCK.unlink()


# ---------------------------------------------------------------------- sync
def sync(cfg, q) -> None:
    """Queue every open PR carrying the approval label; drop withdrawn ones."""
    prs = gh_json("pr", "list", "--repo", cfg["repo"], "--state", "open",
                  "--label", cfg["approve_label"], "--json",
                  "number,headRefOid,baseRefName,isDraft,title")
    approved = {p["number"]: p for p in prs if not p["isDraft"]}
    for n, p in approved.items():
        js = jobs_for(q, n)
        pending = [j for j in js if j["state"] == "pending"]
        if pending:
            pending[0].update(head=p["headRefOid"], base=p["baseRefName"])
        elif not any(j["state"] == "running" for j in js):
            if not any(j.get("head") == p["headRefOid"] for j in js):
                add_job(q, "review", pr=n, head=p["headRefOid"],
                        base=p["baseRefName"], title=p["title"])
                log(f"queued PR #{n} @ {p['headRefOid'][:7]}: {p['title']}")
    for j in q["jobs"]:
        if j["kind"] == "review" and j["state"] == "pending" and j["pr"] not in approved:
            j["state"] = "cancelled"
            j["note"] = "approval label removed"
            log(f"cancelled PR #{j['pr']}: approval withdrawn")
    if cfg["lessons_every_days"]:
        done = [j for j in q["jobs"] if j["kind"] == "lessons"
                and j["state"] in ("pending", "running", "done")]
        last = max((j["added"] for j in done), default=None)
        due = last is None or (dt.datetime.now().astimezone()
                               - dt.datetime.fromisoformat(last)).days >= cfg["lessons_every_days"]
        if due and not any(j["state"] in ("pending", "running") for j in done):
            add_job(q, "lessons")
            log("queued lessons review (scheduled)")


# ------------------------------------------------------------------ capacity
def in_window(window: str) -> bool:
    a, b = window.split("-")
    t = dt.datetime.now().strftime("%H:%M")
    return a <= t < b if a <= b else (t >= a or t < b)


def capacity(cfg, q) -> tuple[bool, str]:
    if not in_window(cfg["window"]):
        return False, f"outside window {cfg['window']}"
    today = dt.date.today().isoformat()
    started = [j for j in q["jobs"] if (j.get("started") or "").startswith(today)]
    if len(started) >= cfg["max_runs_per_day"]:
        return False, f"daily cap reached ({cfg['max_runs_per_day']})"
    load = os.getloadavg()[0] / (os.cpu_count() or 1)
    if load > cfg["max_load_per_cpu"]:
        return False, f"machine busy (load/cpu {load:.2f} > {cfg['max_load_per_cpu']})"
    if cfg["require_ac_power"] and shutil.which("pmset"):
        if "AC Power" not in sh("pmset", "-g", "batt", check=False):
            return False, "on battery"
    return True, "ok"


# ------------------------------------------------------------- preparation
CLONE = STATE / "repo"


def prepare_clone(cfg) -> str:
    if not (CLONE / ".git").exists():
        sh("gh", "repo", "clone", cfg["repo"], str(CLONE), "--", "--quiet")
    sh("git", "-C", str(CLONE), "fetch", "--quiet", "--prune", "origin")
    sh("git", "-C", str(CLONE), "worktree", "prune")
    return sh("gh", "repo", "view", cfg["repo"], "--json", "defaultBranchRef",
              "-q", ".defaultBranchRef.name")


def fresh_worktree(name: str, ref: str, branch: str | None = None) -> Path:
    wt = STATE / "work" / name
    if wt.exists():
        sh("git", "-C", str(CLONE), "worktree", "remove", "--force", str(wt), check=False)
        shutil.rmtree(wt, ignore_errors=True)
    wt.parent.mkdir(parents=True, exist_ok=True)
    if branch:
        sh("git", "-C", str(CLONE), "worktree", "add", "-B", branch, str(wt), ref)
    else:
        sh("git", "-C", str(CLONE), "worktree", "add", "--detach", str(wt), ref)
    return wt


def ensure_skill(wt: Path) -> None:
    """A PR based on a branch without the skill still gets reviewed with it."""
    dst = wt / ".claude" / "skills" / "repo-review"
    if not dst.exists():
        shutil.copytree(REPO_ROOT / ".claude" / "skills" / "repo-review", dst)


def prepare_review(cfg, job) -> Path:
    prepare_clone(cfg)
    n, base = job["pr"], job["base"]
    sh("git", "-C", str(CLONE), "fetch", "--quiet", "origin",
       f"+refs/pull/{n}/head:refs/reviewq/pr{n}")
    head = sh("git", "-C", str(CLONE), "rev-parse", f"refs/reviewq/pr{n}")
    if head != job["head"]:
        raise RuntimeError(f"PR head moved to {head[:7]} after approval of "
                           f"{job['head'][:7]}; approve again to review it")
    wt = fresh_worktree(f"pr{n}", f"origin/{base}")
    p = subprocess.run(["git", "-C", str(wt), "merge", "--no-edit", "--no-ff",
                        "-m", f"reviewq: PR #{n} onto {base}", f"refs/reviewq/pr{n}"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        sh("git", "-C", str(wt), "merge", "--abort", check=False)
        raise RuntimeError(f"PR #{n} doesn't merge cleanly onto {base}:\n{p.stdout}{p.stderr}")
    job["base_sha"] = sh("git", "-C", str(CLONE), "rev-parse", f"origin/{base}")
    ensure_skill(wt)
    return wt


def prepare_lessons(cfg, job) -> Path:
    default = prepare_clone(cfg)
    branch = f"lessons/{dt.date.today().isoformat()}"
    wt = fresh_worktree("lessons", f"origin/{default}", branch=branch)
    job["branch"] = branch
    ensure_skill(wt)
    return wt


# --------------------------------------------------------------------- claude
MEMORY = STATE / "memory" / "MEMORY.md"
PROPOSALS = STATE / "memory" / "proposals"
REPORTS = STATE / "reports"


def reviewer_settings() -> Path:
    tpl = (HERE / "reviewer-settings.json").read_text()
    out = STATE / "reviewer-settings.json"
    out.write_text(tpl.replace("{STATE}", str(STATE).lstrip("/")))
    return out


def system_prompt(cfg, job, wt: Path) -> str:
    tpl = (HERE / "reviewer-prompt.md").read_text()
    memory = MEMORY.read_text().strip() if MEMORY.exists() else "(empty)"
    earlier = sorted(REPORTS.glob(f"pr{job.get('pr')}-*.md")) if job.get("pr") else []
    return tpl.format(
        kind=job["kind"],
        target=(f"PR #{job['pr']} at head {job['head']}, already merged onto "
                f"origin/{job['base']} ({job.get('base_sha', '')[:7]}) in {wt}"
                if job["kind"] == "review" else
                f"branch {job.get('branch')} at the default branch, in {wt}"),
        port=8700 + (job.get("pr") or 0) % 100,
        memory=memory,
        earlier="\n".join(f"- {p}" for p in earlier[-3:]) or "- none",
    )


def run_claude(cfg, job, wt: Path) -> dict:
    prompt = (f"/repo-review review {job['pr']}" if job["kind"] == "review"
              else "/repo-review lessons")
    cmd = [cfg["claude"], "-p", prompt,
           "--permission-mode", "dontAsk",
           "--settings", str(reviewer_settings()),
           "--append-system-prompt", system_prompt(cfg, job, wt),
           "--output-format", "json"]
    if cfg["model"]:
        cmd += ["--model", cfg["model"]]
    logf = STATE / "logs" / f"job{job['id']}.log"
    log(f"job {job['id']}: running claude in {wt} (log {logf})")
    with open(logf, "w") as err:
        proc = subprocess.Popen(cmd, cwd=wt, stdout=subprocess.PIPE, stderr=err,
                                text=True, start_new_session=True)
        try:
            out, _ = proc.communicate(timeout=cfg["timeout_minutes"] * 60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            time.sleep(5)
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            raise RuntimeError(f"timed out after {cfg['timeout_minutes']} min")
    if proc.returncode != 0:
        raise RuntimeError(f"claude exited {proc.returncode}; see {logf}")
    res = json.loads(out)
    if res.get("is_error"):
        raise RuntimeError(f"claude reported an error: {res.get('result', '')[:500]}")
    return res


def save_proposals(job, report: str) -> Path | None:
    m = re.search(r"^##+\s*Proposed memory\s*$(.*?)(?=^##\s|\Z)", report, re.M | re.S)
    body = (m.group(1).strip() if m else "")
    if not body or body.lower().strip(" .-*") in ("none", "n/a"):
        return None
    PROPOSALS.mkdir(parents=True, exist_ok=True)
    path = PROPOSALS / f"job{job['id']}.md"
    path.write_text(f"<!-- from job {job['id']} ({job['kind']} "
                    f"{('PR #' + str(job['pr'])) if job.get('pr') else ''}), {now()} -->\n{body}\n")
    return path


def run_job(cfg, q, job) -> None:
    if not acquire_lock():
        log(f"job {job['id']}: another run holds the lock (pid {lock_holder()})")
        return
    job.update(state="running", started=now(), pid=os.getpid())
    save_queue(q)
    label = f"PR #{job['pr']}" if job["kind"] == "review" else "lessons"
    try:
        wt = prepare_review(cfg, job) if job["kind"] == "review" else prepare_lessons(cfg, job)
        res = run_claude(cfg, job, wt)
        report = res.get("result", "")
        REPORTS.mkdir(parents=True, exist_ok=True)
        name = (f"pr{job['pr']}-{job['head'][:7]}-job{job['id']}.md"
                if job["kind"] == "review" else f"lessons-job{job['id']}.md")
        path = REPORTS / name
        path.write_text(report + "\n")
        prop = save_proposals(job, report)
        verdict = re.search(r"\*\*Verdict\*\*:?\s*(.+)", report)
        job.update(state="done", report=str(path), session=res.get("session_id"),
                   verdict=verdict.group(1).strip() if verdict else None,
                   proposals=str(prop) if prop else None)
        log(f"job {job['id']}: {label} done: {job['verdict']} -> {path}")
        notify(cfg, "reviewq", f"{label}: {job['verdict'] or 'done'}")
    except Exception as e:  # record and move on; the queue keeps going
        job.update(state="failed", note=str(e)[:2000])
        log(f"job {job['id']}: {label} failed: {e}")
        notify(cfg, "reviewq", f"{label} failed: {str(e)[:120]}")
    finally:
        job["finished"] = now()
        job.pop("pid", None)
        if job["kind"] == "review":
            # Approval covers one head: a new push needs a new approval.
            add = ["--add-label", cfg["done_label"]] if job["state"] == "done" else []
            sh("gh", "pr", "edit", str(job["pr"]), "--repo", cfg["repo"],
               "--remove-label", cfg["approve_label"], *add, check=False)
        save_queue(q)
        release_lock()


# ------------------------------------------------------------------ commands
def cmd_setup(cfg, q, a):
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "memory").mkdir(exist_ok=True)
    if not (STATE / "config.json").exists():
        c = {k: v for k, v in DEFAULTS.items() if k != "repo"}
        (STATE / "config.json").write_text(json.dumps(c, indent=2) + "\n")
    if not MEMORY.exists():
        MEMORY.write_text("# Reviewer memory\n\nMaintained by the maintainer. "
                          "Reviews propose additions; `reviewq memory accept` adds them.\n")
    for name, color, desc in [
        (cfg["approve_label"], "1d76db", "Approved for a local review (reviewq)"),
        (cfg["done_label"], "0e8a16", "Local review done (reviewq)"),
        ("first-pass", "c5def5", "Run the GitHub first-pass review again"),
    ]:
        sh("gh", "label", "create", name, "--repo", cfg["repo"], "--color", color,
           "--description", desc, "--force")
    prepare_clone(cfg)
    print(f"ready: state in {STATE}, labels {cfg['approve_label']}/{cfg['done_label']}")


def cmd_approve(cfg, q, a):
    sh("gh", "pr", "edit", str(a.pr), "--repo", cfg["repo"], "--add-label", cfg["approve_label"])
    sync(cfg, q)
    save_queue(q)


def cmd_queue(cfg, q, a):
    rows = q["jobs"] if a.all else [j for j in q["jobs"] if j["state"] in ("pending", "running")
                                    or j.get("finished", "") >= (dt.date.today() - dt.timedelta(days=7)).isoformat()]
    if not rows:
        print("queue empty")
    for j in rows:
        what = f"PR #{j['pr']} @ {j['head'][:7]}" if j["kind"] == "review" else "lessons"
        extra = j.get("verdict") or j.get("note", "")
        print(f"{j['id']:>4} {j['state']:<9} {what:<20} {j.get('added', '')[:16]}  {extra[:80]}")
    ok, why = capacity(cfg, q)
    holder = lock_holder()
    print(f"\ncapacity: {why}; running: {'pid ' + str(holder) if holder else 'no'}")


def cmd_sync(cfg, q, a):
    sync(cfg, q)
    save_queue(q)


def cmd_tick(cfg, q, a):
    sync(cfg, q)
    save_queue(q)
    if lock_holder():
        return
    pending = [j for j in q["jobs"] if j["state"] == "pending"]
    if not pending:
        return
    ok, why = capacity(cfg, q)
    if not ok:
        log(f"tick: {len(pending)} pending, waiting: {why}")
        return
    if a.dry_run:
        print(f"would run job {pending[0]['id']}")
        return
    run_job(cfg, q, pending[0])


def cmd_run(cfg, q, a):
    """Run now, ignoring the window/cap/load (still one at a time)."""
    sync(cfg, q)
    if a.pr:
        job = next((j for j in q["jobs"] if j.get("pr") == a.pr and j["state"] == "pending"), None)
        if not job:
            sys.exit(f"PR #{a.pr} isn't queued: approve it first (reviewq approve {a.pr})")
    else:
        job = next((j for j in q["jobs"] if j["state"] == "pending"), None)
        if not job:
            sys.exit("nothing pending")
    if a.prepare_only:
        wt = prepare_review(cfg, job) if job["kind"] == "review" else prepare_lessons(cfg, job)
        save_queue(q)
        print(f"prepared {wt}\n\n--- system prompt ---\n{system_prompt(cfg, job, wt)}")
        return
    run_job(cfg, q, job)


def cmd_cancel(cfg, q, a):
    for j in jobs_for(q, a.pr):
        if j["state"] == "pending":
            j.update(state="cancelled", note="cancelled by hand")
    sh("gh", "pr", "edit", str(a.pr), "--repo", cfg["repo"],
       "--remove-label", cfg["approve_label"], check=False)
    save_queue(q)


def latest_job(q, pr):
    done = [j for j in jobs_for(q, pr) if j.get("report")]
    if not done:
        sys.exit(f"no report for PR #{pr}")
    return done[-1]


def cmd_report(cfg, q, a):
    j = latest_job(q, a.pr)
    print(Path(j["report"]).read_text())
    if j.get("session"):
        print(f"\n(ask follow-ups: cd {STATE / 'work' / ('pr' + str(a.pr))} && "
              f"claude --resume {j['session']})")


def cmd_post(cfg, q, a):
    j = latest_job(q, a.pr)
    text = Path(j["report"]).read_text()
    text = re.split(r"^##+\s*Proposed memory\s*$", text, flags=re.M)[0].rstrip()
    print(text)
    if not a.yes and input(f"\nPost this on PR #{a.pr}? [y/N] ").strip().lower() != "y":
        return
    body = STATE / "post.md"
    body.write_text(text + "\n\n<sub>Local review by reviewq (/repo-review).</sub>\n")
    sh("gh", "pr", "comment", str(a.pr), "--repo", cfg["repo"], "--body-file", str(body))
    print("posted")


def cmd_memory(cfg, q, a):
    PROPOSALS.mkdir(parents=True, exist_ok=True)
    if a.action == "show":
        print(MEMORY.read_text() if MEMORY.exists() else "(empty)")
    elif a.action == "proposals":
        files = sorted(PROPOSALS.glob("*.md"))
        if not files:
            print("no proposals")
        for f in files:
            print(f"=== {f.stem}\n{f.read_text()}")
    elif a.action in ("accept", "reject"):
        f = PROPOSALS / f"{a.id}.md"
        if not f.exists():
            sys.exit(f"no proposal {a.id} (see: reviewq memory proposals)")
        if a.action == "accept":
            text = f.read_text()
            if a.edit:
                subprocess.run([os.environ.get("EDITOR", "vi"), str(f)])
                text = f.read_text()
            with open(MEMORY, "a") as m:
                m.write(f"\n## {dt.date.today().isoformat()} ({a.id})\n"
                        + re.sub(r"<!--.*?-->\n?", "", text).strip() + "\n")
            print(f"added to {MEMORY}")
        f.unlink()


def cmd_lessons(cfg, q, a):
    add_job(q, "lessons")
    save_queue(q)
    print("lessons review queued")


def cmd_install(cfg, q, a):
    plist = Path.home() / "Library" / "LaunchAgents" / f"{LABEL_PLIST}.plist"
    path_dirs = []
    for tool in ("claude", "gh", "uv", "uvx", "git"):
        w = shutil.which(cfg["claude"] if tool == "claude" else tool)
        if w and os.path.dirname(w) not in path_dirs:
            path_dirs.append(os.path.dirname(w))
    path_dirs += ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]
    (STATE / "logs").mkdir(parents=True, exist_ok=True)
    plist.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>{LABEL_PLIST}</string>
  <key>ProgramArguments</key><array>
    <string>/usr/bin/python3</string><string>{HERE / 'reviewq.py'}</string><string>tick</string>
  </array>
  <key>StartInterval</key><integer>{int(cfg['tick_minutes']) * 60}</integer>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>{':'.join(path_dirs)}</string>
    <key>HOME</key><string>{Path.home()}</string>
    <key>REVIEWQ_HOME</key><string>{STATE}</string>
  </dict>
  <key>StandardOutPath</key><string>{STATE / 'logs' / 'launchd.log'}</string>
  <key>StandardErrorPath</key><string>{STATE / 'logs' / 'launchd.log'}</string>
</dict></plist>
""")
    uid = os.getuid()
    subprocess.run(["launchctl", "bootout", f"gui/{uid}", str(plist)], capture_output=True)
    sh("launchctl", "bootstrap", f"gui/{uid}", str(plist))
    print(f"installed {plist}: tick every {cfg['tick_minutes']} min")


def cmd_uninstall(cfg, q, a):
    plist = Path.home() / "Library" / "LaunchAgents" / f"{LABEL_PLIST}.plist"
    subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}", str(plist)], capture_output=True)
    plist.unlink(missing_ok=True)
    print("uninstalled")


def main():
    ap = argparse.ArgumentParser(prog="reviewq", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup", help="create state dir, config, memory, labels, clone")
    p = sub.add_parser("approve", help="approve a PR for local review")
    p.add_argument("pr", type=int)
    p = sub.add_parser("queue", help="show the queue and capacity")
    p.add_argument("--all", action="store_true")
    sub.add_parser("sync", help="queue approved PRs from GitHub")
    p = sub.add_parser("tick", help="sync, then start one job if there is capacity")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("run", help="run a queued job now, ignoring capacity")
    p.add_argument("pr", type=int, nargs="?")
    p.add_argument("--prepare-only", action="store_true",
                   help="prepare the worktree and print the prompt; don't run claude")
    p = sub.add_parser("cancel", help="withdraw a PR's approval")
    p.add_argument("pr", type=int)
    p = sub.add_parser("report", help="print the latest report for a PR")
    p.add_argument("pr", type=int)
    p = sub.add_parser("post", help="post the latest report on the PR (asks first)")
    p.add_argument("pr", type=int)
    p.add_argument("--yes", action="store_true")
    p = sub.add_parser("memory", help="show memory; list, accept or reject proposals")
    p.add_argument("action", choices=["show", "proposals", "accept", "reject"])
    p.add_argument("id", nargs="?")
    p.add_argument("--edit", action="store_true", help="edit a proposal before accepting")
    sub.add_parser("lessons", help="queue a lessons review")
    sub.add_parser("install", help="install the launchd agent that runs tick")
    sub.add_parser("uninstall", help="remove the launchd agent")
    a = ap.parse_args()
    cfg = load_config()
    q = load_queue()
    globals()[f"cmd_{a.cmd}"](cfg, q, a)


if __name__ == "__main__":
    main()
