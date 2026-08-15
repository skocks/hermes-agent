---
name: pi
description: "Delegate coding to the Pi coding agent CLI; visible in a Herdr tab when available."
version: 0.2.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Coding-Agent, Pi, Herdr, Automation]
    related_skills: [claude-code, codex, opencode, hermes-agent]
---

# Pi — Hermes Orchestration Guide

Delegate coding tasks to the Pi CLI. CLI flags below are assumed by analogy
with Claude Code/Codex/OpenCode (unverified) — check `pi --help` once and
fix this file if wrong.

## When to Use

User asks for Pi, or for coding work with no agent named and Pi is the
configured/preferred coding agent.

## Step 0 — Detect Herdr

```bash
test "${HERDR_ENV:-}" = 1 && echo herdr || echo no-herdr
```

Check every task, not just once per session.

## Herdr Mode (preferred — visible to the user)

Reuse a live `pi-work` agent if one exists (`herdr agent list`); otherwise
create it once:

```bash
herdr tab create --workspace "$HERDR_WORKSPACE_ID" --cwd "<workdir>" --label pi --no-focus
# -> .result.root_pane.pane_id
herdr pane run <root_pane_id> "pi"
herdr wait agent-status <root_pane_id> --status idle --timeout 60000
herdr agent rename <root_pane_id> pi-work
```

Send the task and submit it (send does not press Enter):

```bash
herdr agent send pi-work "<task>"
herdr pane send-keys pi-work enter
herdr agent wait pi-work --status idle --timeout 300000
```

`blocked` means Pi is asking something — `herdr agent read pi-work
--source recent-unwrapped --lines 150`, answer it, wait again; don't treat
it as done. Keep the tab open across tasks. `--no-focus` is fine — the tab
stays inspectable without stealing the user's view.

## No-Herdr Mode

Herdr's absence never blocks starting Pi — just shell out directly:

```bash
terminal(command="tmux new-session -d -s pi-work -x 140 -y 40")
terminal(command="tmux send-keys -t pi-work 'cd <workdir> && pi' Enter")
terminal(command="sleep 2 && tmux send-keys -t pi-work '<task>' Enter")
terminal(command="sleep 20 && tmux capture-pane -t pi-work -p -S -60")
```

Kill the session when done.

## Pitfalls

- New `pi-work` tab per task instead of reusing the idle one — litters tabs.
- Treating `blocked` as finished.
- Trusting the unverified flags above without checking `pi --help` once.

## Verification

- [ ] Checked `HERDR_ENV` this task
- [ ] Reused existing `pi-work` agent when idle, didn't duplicate
- [ ] Task actually submitted (send + enter)
- [ ] Waited out any `blocked` state
- [ ] No stray empty panes or dangling tmux sessions
