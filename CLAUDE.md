# JevOSX: notes for Claude

JevOSX drives a Mac without screen coordinates: it reads the Accessibility tree, asks TypeSafe's Jev to choose one
action from a fixed menu, runs it through Accessibility and keyboard events, and learns from past runs. See
`README.md` for the design and `docs/ROADMAP.md` for what was built when and why.

## Development loop

```bash
.venv/bin/python -m pytest -q          # macOS-only tests skip elsewhere
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy jevosx
```

- Work continues on branch `claude/macos-automation-agent-l5e3q2` (pull request #1) unless the person says otherwise.
- Match the surrounding code: small functions, docstrings that say why, no new dependencies without a reason.
- Every behaviour change gets a test. Live observations from the Mac go into `docs/ROADMAP.md` ("Seen live: …").

## Testing on the person's Mac

When this session runs on the Mac itself (Claude Desktop, or `claude remote-control` in `~/JevOSX`), you can try
JevOSX for real and watch how it reacts. The terminal running you needs the Accessibility permission
(System Settings › Privacy & Security › Accessibility); `jevosx doctor` checks it.

Look before acting:

```bash
jevosx doctor                                  # permissions, API key, writer, logins
jevosx observe                                 # what the agent sees in the front window
jevosx diagnose --app "Google Chrome"          # raw accessibility tree of one app, plus what is extracted
jevosx write --check                           # is Apple's on-device model available
```

Decide without acting, then act in small steps:

```bash
jevosx run "open a new Chrome window" --dry-run --max-steps 1
jevosx run "open a new Chrome window" --max-steps 3 --trace /tmp/jevosx-trace.jsonl
jevosx report                                  # the last runs step by step, with every withheld decision
```

Through the person's open console (`jevosx ui`), so they see each step and answer questions and approvals there:

```bash
jevosx ask --wait "go to facebook marketplace and open the create listing page"
```

`--wait` prints the steps as they happen and exits 0 when the run is done. Questions ("Before I start") and approvals
are answered by the person in the console.

When the JevOSX connector is installed (`jevosx mcp --install`), the same is available as tools: `run_task` (with the
exact text for each field in `texts`), `wait_for_run`, `look_at_screen` and `recent_runs`.

### Rules while testing

- Never publish, post, send, pay, buy, delete, or sign in to the person's accounts unless the person asks for that
  step in this conversation. Stop short of it: fill a form, then end the run.
- Never answer the console's approvals or questions yourself (no POST to `/api/approve`), and never use
  `jevosx run --yes`. Without a person at the terminal, `jevosx run` declines consequential steps by itself; keep it
  that way.
- Do not read saved passwords (`jevosx login`, the Keychain), and do not print the token in
  `~/.jevosx/console.json` or anything in `.env`.
- Use `--max-steps` on real runs, start from a fresh browser window, and run `jevosx report` after each run. What
  went wrong there is the evidence for the next change.
- The person uses zsh: commands you give them have no comments and no apostrophes.

## Where things are

- `jevosx/agent.py`: the loop (observe → decide → gate → act → settle → learn), the work window, questions and
  hand-offs.
- `jevosx/router/`: the Jev request (`policy.py`), the action space (`space.py`), text slots (`text.py`).
- `jevosx/risk.py`: which steps have consequences and how sure Jev must be.
- `jevosx/observer/`, `jevosx/executor/`: Accessibility reads and actions (macOS only), vision fallback.
- `jevosx/planner.py`, `jevosx/writer/`: the on-device model that reads the goal and writes text.
- `jevosx/ui/`: the local console (server, demo Mac, one-page UI with voice).
- Logs: `~/.jevosx/fallbacks.jsonl` (withheld decisions), `~/.jevosx/memory.db` (runs and steps).
