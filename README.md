# JevOSX

**A coordinate-free macOS automation agent.** JevOSX reads the native Accessibility tree
(`AXUIElement`) of whatever is on screen, turns it into a compact indexed text map, and asks
[TypeSafe's Jev](https://docs.typesafe.ai/introduction), a "System One" decision model, to choose the next
operation and its target from a fixed menu of options. It then runs that choice deterministically through
Accessibility actions and keyboard events. Every run is saved to a local SQLite memory. Similar past runs
come back as hints, so repeated tasks get better over time.

The model never outputs coordinates, selectors, shell commands or free text. It only picks ids that the agent
observed on this Mac. The Accessibility tree comes first. Only for apps that draw their own interface (games, canvas
design tools) does JevOSX capture that one window and read its text with Apple's on-device OCR, and even then Jev
picks recognized text by id while the click point is computed locally.

It can also:
- write new text on the Mac (Apple's on-device model), for example "write a poem about autumn in TextEdit";
- sign in to websites with logins kept in the macOS Keychain;
- hand a step to you (a 2FA code, a CAPTCHA) and carry on afterwards.

See [docs/ROADMAP.md](docs/ROADMAP.md) for the research behind these and what comes next.

Inspired by [jev-ultrafast](https://github.com/browser-use/jev-ultrafast) (browser) and typesafe-computer-use
(OCR-based desktop). JevOSX brings their typed-choice loop to the whole Mac, with accessibility data first and OCR as the fallback.

---

## Contents

- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [What Jev sees](#what-jev-sees)
- [Guarantees and guardrails](#guarantees-and-guardrails)
- [Setup](#setup)
- [Usage](#usage)
- [Memory and the learning loop](#memory-and-the-learning-loop)
- [Configuration](#configuration)
- [Performance](#performance)
- [Limitations](#limitations)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## How it works

```mermaid
flowchart LR
    subgraph Mac[Your Mac: everything here stays local]
        O[Observer<br/>AXUIElement tree walk] --> S[Indexed text map<br/>elements · menus · apps · windows]
        S --> M[(Memory<br/>SQLite trajectories)]
        M -->|similar past steps| H[Hints]
        S --> R[Jev router<br/>action space + choice questions]
        H --> R
        G{Confidence gate<br/>≥ 0.65?} -->|yes| F[Safety + freshness checks]
        G -->|no| FB[Fallback: retry · ask · stop<br/>+ fallback log]
        F --> E[Executor<br/>AXPress · AXValue · menus · keys]
        E --> O
        E -->|step + outcome| M
    end
    R -->|one request: state + choice questions| J[TypeSafe Jev]
    J -->|choice + probabilities + confidence| G
```

One decision cycle:

1. **Observe.** The frontmost app's focused window is walked through the Accessibility API. Attribute reads are
   batched, one IPC round trip per element. Hidden, zero-size, off-screen and disabled elements are dropped.
   What remains becomes an indexed table such as `[3] textfield "To" = ""`. Menu-bar commands, other windows and
   running apps are collected as whole-Mac context.
2. **Recall.** Local memory finds finished runs with similar goals and similar screens. It maps what worked before
   (or had no effect) onto ids that are on screen now.
3. **Decide.** One Jev request carries the state plus a set of discrete choice questions: which
   **operation** (`CLICK`, `TYPE_TEXT`, `MENU`, `PRESS_KEY`, `SCROLL_*`, `OPEN_APP`, `FOCUS_WINDOW`, `ASK_USER`,
   `WAIT`, `DONE`, `BLOCKED`), and, speculatively in the same round trip, the **target** for each operation. Only the target head
   of the chosen operation is validated and used.
4. **Gate.** If Jev's confidence in the operation or in the chosen target is below the floor (0.65 by default),
   nothing runs. The fallback policy re-observes, asks you, or stops, and the decision is written to a fallback log.
5. **Guard and act.** Safety rules run next (deny lists, confirmation for consequential labels such as *Delete* or
   *Buy*, password-field rules). The target handle is re-validated, then executed with `AXPress`, an `AXValue`
   write, a menu-item press, `AXRaise`, a scroll-bar value change or a CGEvent key chord.
6. **Settle and learn.** The loop waits for a cheap UI signature to stop changing, re-observes, labels the step
   *changed* or *unchanged*, and stores it. When the run ends, its status (optionally confirmed by a verifier or by
   you) decides how strongly it counts in future retrieval.

## Repository layout

```
jevosx/
├── observer/            # OS observer: Accessibility tree → Observation
│   ├── ax.py            #   pyobjc AXUIElement wrapper (batched reads, timeouts, typed errors)
│   ├── walker.py        #   bounded tree walk: pruning, labels, rows, visibility clip (platform-independent)
│   ├── menus.py         #   menu-bar walker → MENU targets with paths and shortcuts
│   ├── apps.py          #   running apps (NSWorkspace) + installed app bundles (Info.plist)
│   ├── desktop.py       #   MacDesktopObserver: frontmost app/window, web accessibility, caching
│   ├── vision.py        #   OCR fallback for apps that draw their own interface (window capture + Vision)
│   └── textmap.py       #   human-readable text map (`jevosx observe`)
├── router/              # Jev router
│   ├── client.py        #   HTTP/2 Jev client, retries, strict choice-only contract, answer validation
│   ├── space.py         #   dynamic action space: operations and compatible targets per observation
│   ├── policy.py        #   state building, context budget, request contract, decision decoding
│   ├── prompts.py       #   instructions sent with each question
│   └── text.py          #   TYPE_TEXT sources: text slots, goal literals, GENERATE via the writer
├── writer/              # Free-form text ("write a poem"): Jev picks the field, a writer fills in the words
│   ├── apple.py         #   Apple's on-device model via a compiled Swift helper (build, check, requests)
│   ├── apple_writer.swift  # the helper: Foundation Models over JSON stdin/stdout
│   ├── openai.py        #   optional OpenAI-compatible chat model
│   ├── base.py          #   when to offer writing, prompts, output cleanup
│   └── simulated.py     #   canned writer for demo mode
├── executor/            # Execution engine
│   ├── mac.py           #   MacExecutor: AX actions, value writes, scrolling, app activation/launch
│   ├── input.py         #   CGEvent keyboard (layout-independent Unicode) and opt-in pointer fallback
│   ├── keys.py          #   named key-chord vocabulary for PRESS_KEY
│   ├── safety.py        #   deny/confirm policy
│   └── base.py          #   Executor protocol and DryRunExecutor
├── memory/              # Local memory and learning layer
│   ├── store.py         #   SQLite trajectories (WAL, versioned schema, JSONL export/import, prune)
│   ├── embedding.py     #   deterministic hashed n-gram embeddings (numpy, no downloads)
│   └── retriever.py     #   similarity retrieval → actionable hints mapped to current ids
├── logins.py            # saved website logins: Keychain storage, site-bound username/password slots
├── sites.py             # host names, https pages and which hosts a saved login may be used on
├── agent.py             # the loop: observe → recall → decide → gate → guard → act → settle → learn
├── config.py            # typed settings (TOML + .env + environment), strict validation
├── types.py             # shared data model (UIElement, Observation, Action, …)
├── ui/                  # local web console (`jevosx ui`)
│   ├── server.py        #   stdlib HTTP + Server-Sent Events, run manager, approvals, token/Host guards
│   ├── demo.py          #   simulated Mac + simulated decisions for `--demo`
│   └── static/          #   single-file front end (no external resources)
└── cli.py               # `jevosx run | observe | ui | write | diagnose | doctor | report | memory`
docs/                    # ROADMAP.md: research notes, phases and risks
examples/                # run_agent.py (Python API), dump_tree.py (inspect what the agent sees)
config/                  # jevosx.example.toml: every setting with its default
scripts/                 # install.sh (one-line installer), bootstrap.sh (venv + install + doctor)
tests/                   # offline tests: fake AX trees, fake desktop, mocked Jev endpoint
```

## What Jev sees

The request goes to `POST https://api.typesafe.ai/v1/systemone` with `{model, state, questions}`. The state
excerpt below is for a TextEdit window (`jevosx observe --json` prints the real one for any app):

```json
{
  "desktop": {"frontmost_app": "TextEdit", "bundle_id": "com.apple.TextEdit", "window": "Untitled",
              "focused_element": "[4] textarea \"Body\""},
  "elements": [
    {"index": 1, "role": "close button", "label": "close button", "ops": ["CLICK"]},
    {"index": 2, "role": "popup", "label": "Paragraph Styles", "value": "Body", "in": "toolbar", "ops": ["CLICK"]},
    {"index": 4, "role": "textarea", "label": "Body", "state": ["focused"], "ops": ["TYPE_TEXT", "CLICK"]}
  ],
  "visible_text": "…",
  "recent_actions": [{"step": 1, "action": "MENU [m2] File › New (⌘N)", "result": "ui changed"}],
  "memory_hints": [{"kind": "worked", "operation": "TYPE_TEXT", "target_id": "4", "past_runs": 3, "similarity": 0.82}],
  "text_slots": {"quote_1": "Hello from JevOSX", "email": "me@example.com"}
}
```

Every question is a discrete one-of-N `choice`:

```json
{
  "operation":      {"type": "choice", "criteria": {"CLICK": "…", "TYPE_TEXT": "…", "MENU": "…", "PRESS_KEY": "…",
                                                    "OPEN_APP": "…", "WAIT": "…", "DONE": "…", "BLOCKED": "…"},
                     "instructions": {"goal": "…", "rules": "…"}},
  "click_target":   {"type": "choice", "criteria": {"1": {"element": "[1] close button …"}, "2": {…}, "4": {…}}},
  "menu_target":    {"type": "choice", "criteria": {"m1": {"command": "File › New", "shortcut": "⌘N"}, …}},
  "key_target":     {"type": "choice", "criteria": {"RETURN": {…}, "CMD_S": {…}, …}},
  "text_slot":      {"type": "choice", "criteria": {"quote_1": {…}, "email": {…}}}
}
```

Jev answers each question with `{choice, probabilities (one per id), confidence}`.

**What leaves the Mac, and what doesn't:**

| Sent to TypeSafe | Stays local |
| --- | --- |
| Goal, instructions | AX handles, element frames/coordinates |
| Roles, labels, non-secret values, states of interactive elements | Window captures for OCR (read on the Mac, then deleted) |
| Visible static text (capped, 2,000 chars by default) | Password-field values (never read) |
| Menu command paths, app and window names, page address (no query string) | Secret text-slot values and saved passwords (sent as `••••••`) |
| Recent action descriptions, memory hints, text recognized on screen (OCR) | The memory database, fallback log and traces; everything the on-device writer sees |

## Guarantees and guardrails

**Strict choice schemas.** The client refuses to send anything that is not a `type: "choice"` question with 2–255
ids (`RouterContractError`). Before each request, `validate_request` checks three things. The operation options
equal the action space. Every `click_target` / `type_text_target` id is an `index` in the element table sent
with it. Every offered id resolves to a handle observed on this Mac. Answers are validated as a real distribution
over exactly the offered ids, and the choice must be the argmax. A malformed answer on the selected head raises
`JevResponseError` and nothing is executed. A malformed answer on an unused speculative head is ignored.

**Context guardrails.** These run before anything is serialized:
- subtrees outside the window or scroll-area clip are pruned, as are `AXHidden` subtrees and window chrome;
- zero-size elements are never indexed;
- read-only text fields become text, not controls;
- disabled and non-interactive elements are left out of the element table (`jev.include_disabled = false`);
- installed apps that aren't running are offered to `OPEN_APP` only when the goal names them;
- the serialized state has a hard budget (`jev.max_state_bytes`, 48 KB). Visible text is trimmed first, then
  history, then trailing elements. The focused element is never dropped, and dropped elements are removed from
  the target questions too, so ids and the table always match.

The walk itself is bounded by node count, element count, depth, children per node and wall time.

**Confidence gate and fallback.** Each step's floor is compared against the *weakest* confidence among the
answers that would drive execution: operation, chosen target, and text slot. How high the floor is depends on what
a wrong step would cost (`jevosx/risk.py`):

| Tier | Steps | Floor |
| --- | --- | --- |
| safe | open or switch apps and windows, new window or tab, scroll, put the cursor in a field, type into a single-line or empty field, navigation keys (`CMD_L`, `TAB`, arrows…), hand a step to you | `agent.safe_confidence`, **0.2** |
| routine | clicks on buttons, links, checkboxes and list options, `RETURN`, menu commands, replacing text that is already there, `DONE` | `agent.routine_confidence`, **0.3** |
| careful | steps with consequences: anything the safety policy wants confirmed (Delete, Send, Publish, Post, Pay…), `CMD_W`, `CMD_Q`, and a second sign-in attempt in a run (failed logins can lock an account). These also ask you. | `agent.min_confidence`, **0.65** |

A wrong safe or routine step costs a step, which the agent notices and corrects; a wrong careful step is a post you
have to take down. So only careful steps are held back hard.

No tier's floor is ever above `agent.min_confidence`, so raising it (the console's *Confidence floor*, or
`--min-confidence`) makes every step more careful. A step that matches one that worked in a similar earlier run
(a memory hint) counts as safe, unless it is careful: a step you approved once is not asked about again. `DONE`
is gated too, so an unsure `DONE` cannot end a run early. When you approve an unsure step, a consequential click is
not asked about a second time, but typing still is.

One exception: while the web console's own browser window is in front, the agent may only open a new window or
tab, or switch apps or windows. Those moves change nothing, so they are not gated by default
(`agent.gate_console_navigation = true` gates them too). Below the floor, `LowConfidenceError` is raised and
handled:

| `agent.low_confidence_policy` | Behaviour |
| --- | --- |
| `retry` (default) | withhold, wait, re-observe; after `max_low_confidence_retries`, stop with status `low_confidence` |
| `ask` | look again once (`agent.ask_after_retries`), then show you the proposed action; execute only if you approve, otherwise retry |
| `stop` | end the run immediately with status `low_confidence` |

Every withheld decision is appended to `agent.fallback_log` (`~/.jevosx/fallbacks.jsonl`) with the goal, app,
window, top options, confidence, floor, risk tier and resolution. You can also pass your own handler:
`Agent(..., on_low_confidence=lambda exc, obs: "retry" | "execute" | "stop")`.

**Safety and freshness.**
- Deny-listed apps (Keychain Access and Passwords by default) are never operated.
- Labels matching consequential patterns (delete, erase, trash, buy, pay, transfer, shut down, …) and `CMD_Q` need
  confirmation. When there is no terminal to confirm in, they are declined.
- Password fields accept only secret text slots, and a writer is never used for them.
- Saved logins: a password is typed only into a password field on its own site over https. Before typing, the
  executor re-reads the field's page URL. Saved passwords never reach Jev, the writer, memory, logs or history,
  and usernames are masked in all of them.
- The Apple menu is never offered.
- Right before execution the target is re-validated (same role, still enabled). A stale target is re-observed,
  never guessed. Key presses are only sent once the work window is confirmed in front.
- Three consecutive actions with no visible change stop the run as `blocked`.
- A verifier (`--expect-text`) can reject a premature `DONE`.
- `--dry-run` decides without touching the Mac.

## Setup

**Requirements:** macOS (13 Ventura or newer recommended), Python 3.11+, and a TypeSafe API key.

**One-line install.** Open Terminal and paste:

```bash
curl -fsSL https://raw.githubusercontent.com/matthewagi/JevOSX/main/scripts/install.sh | bash
```

What the installer does, without needing `sudo`:
1. Finds Python 3.11+ (or gets Python 3.12 through [uv](https://docs.astral.sh/uv/) if you don't have it).
2. Downloads JevOSX to `~/JevOSX` and installs its dependencies.
3. Asks for your TypeSafe key (hidden input, stored in `~/JevOSX/.env` with `chmod 600`).
4. Opens the Accessibility settings pane.
5. Adds a `jevosx` command and an optional double-clickable `JevOSX.command` on your Desktop.
6. Offers to start the console.

Run it again to update. Overrides: `JEVOSX_DIR`, `JEVOSX_REF`, `JEVOSX_PYTHON`, and `JEVOSX_YES=1` to accept
the defaults without questions.

Or step by step:

```bash
git clone https://github.com/matthewagi/JevOSX.git && cd JevOSX
scripts/bootstrap.sh              # creates .venv, installs deps (incl. pyobjc), copies .env.example → .env, runs doctor
$EDITOR .env                      # set TYPESAFE_API_KEY=...
source .venv/bin/activate
jevosx doctor --live              # checks permissions and measures a real Jev round trip
jevosx ui                         # open the console and start giving commands
```

Manual install:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -e .      # add -r requirements-dev.txt for tests/linting
export TYPESAFE_API_KEY=...
```

**Accessibility permission (required).** macOS only lets trusted processes read other apps' UI and send input.
The first `jevosx doctor`/`observe`/`run` triggers the system prompt. Enable the app that runs Python
(Terminal, iTerm, VS Code, …) under **System Settings › Privacy & Security › Accessibility**, then restart that
app.

**Screen Recording permission (optional).** Only needed for apps that draw their own interface (games, canvas design
tools), whose windows are read by on-device OCR. `jevosx doctor` asks for it. Everything else works without it.

**Browsers and Electron apps.** JevOSX sets `AXManualAccessibility` (Electron: Slack, VS Code, Discord, Notion…)
and `AXEnhancedUserInterface` (Chrome, Edge, Brave, Arc…) so their web content shows up in the tree. Safari
exposes web content natively. The first observation after an app launches can be sparse while the tree builds.

## Usage

### Console (web UI)

```bash
jevosx ui            # opens http://127.0.0.1:8765 in your browser and drives your real Mac
jevosx ui --demo     # simulated Mac + simulated decisions: try the whole loop on any OS, no API key
```

A chat-style console for giving the agent commands:

- **Command box.** Type what you want in plain English and press Enter. The options button sets the start app,
  success text, step budget, confidence floor, what happens when Jev is unsure (retry / ask me / stop), dry run,
  memory use, and text slots.
- **Live step stream.** Every decision appears as it happens: operation, target, whether it ran and how, a
  confidence meter with the floor marked on it, timings (Jev, observe, recall, act, settle), and memory hints.
  Expand *details* to see Jev's top alternatives for the operation and target.
- **Approvals.** Consequential steps (e.g. *Delete*) and, with "Ask me", low-confidence steps pause with
  **Allow / Deny** buttons. **Stop** (or ⌘.) interrupts after the current step.
- **Side panel.** *Screen* shows exactly what Jev sees: the indexed interactive elements, menu commands and visible
  text. *Memory* lists past runs; mark them ✓/✗ to teach retrieval. *Log* is a raw event stream.
- **Security.** It listens on 127.0.0.1 only. Every API call needs a per-session token, and requests with a
  foreign `Host` are rejected, so other web pages can't drive your Mac through it.

**It works behind your window** (`agent.background = true`, the default). When one of the agent's own actions puts
a window in front (a new browser window, an app it opened), that window becomes its work window. From then on the
agent reads that window wherever it is, so you can keep the console or Terminal in front while it works:

- Clicks and field writes go through Accessibility and reach the work window without bringing it forward.
- Key presses, menu commands and typing into browsers need the keyboard. For those the work window comes forward
  for a moment. JevOSX checks that it really is the key window before sending anything, and then brings back the
  window you were using. If it cannot bring the window forward, nothing is sent.
- A window you switch to yourself never becomes work, and the console window never does either. Clicking Allow in
  the console no longer sends the next keystrokes to the console.
- For a "Your turn" hand-off, the work window is brought forward so you can do your part.

macOS has no public way to give the agent a desktop of its own, so a work window that needs typing still flashes
forward briefly. Put the console beside the browser, or on another display, to watch both. Set
`background = false` to have the agent work in front, as before.

Demo mode simulates Finder, TextEdit, Safari and Notes, with a transparent keyword policy standing in for Jev.
The router contract, confidence gate, safety approvals, memory and streaming are the real code, so it behaves
exactly like a live run apart from who makes the decisions. The example cards cover a multi-step task, a web
search, write-and-save, a step needing approval, and a vague request that the confidence gate withholds.

### CLI

```bash
# See exactly what the agent sees (no API key needed). Switch to the target app within 3 s.
jevosx observe --delay 3
jevosx observe --delay 3 --menus
jevosx observe --delay 3 --json > state.json      # the literal Jev state + choice questions (+ size on stderr)

# Pursue a goal. Quoted text in the goal becomes typeable text (no text model needed).
jevosx run 'Create a new document and type "Meeting notes for Monday"' --app TextEdit --expect-text "Meeting notes"

# Provide named text slots for forms (secret-looking names are masked and allowed into password fields).
jevosx run "Fill in the sign-up form and submit it" --slot email=me@example.com --slot password=hunter2

# Inspect a decision without executing anything
jevosx run "Open System Settings and show the Wi-Fi pane" --dry-run

# Tighter supervision
jevosx run "…" --step                                  # approve every action
jevosx run "…" --min-confidence 0.8 --on-low-confidence ask
jevosx run "…" --trace run.jsonl --feedback            # JSONL trace; label the run for memory at the end
```

Exit codes: `0` success/done, `1` blocked, failed, low confidence or out of steps, `2` configuration or platform error.

### Writing new text

Jev only chooses; it never writes. When a goal asks for text that is not in it ("write a poem about autumn in
TextEdit", "reply to Anna saying I'm late", "summarize this page in a new note"), JevOSX offers one more text option,
`GENERATE`. If Jev picks it for the field it chose, a *writer* composes only that field's content:

- **Apple's on-device model** (the default on macOS 26 with Apple Intelligence turned on). It is free and private,
  and it works offline. JevOSX compiles a ~150-line Swift helper once with the Command Line Tools into
  `~/.jevosx/bin/`; Apple's Python SDK would need the full Xcode.
- **Any OpenAI-compatible model** set in `[text_model]`, for Intel Macs or older macOS.

```bash
jevosx run "Open TextEdit and write a short poem about autumn"
jevosx write "a haiku about the sea"     # try the writer on its own
jevosx write --check                     # is Apple's model ready? If not, it says why and how to fix it
```

The writer never fills password fields. It gets screen text as context only, and it composes each field once per
run, so a retry types the same text instead of a new poem. Goals that name the text (`type "hello"`) never use it.

**Reading the request.** At the start of a run the same on-device model reads the whole request once
(`agent.plan = "auto"`). It returns the steps and the exact values to type. For "go to facebook and prepare a
product to sell on marketplace a plastic welding gun for 40 euros generic text" that is:

- website `facebook.com/marketplace`;
- title `Plastic welding gun`;
- price `40`;
- a ready-written description.

The values become text slots that Jev can pick for the fields it chooses. The steps are a hint for ordering. Both
show up as step 0 (`PLAN`) in the terminal and console. Neither ever becomes an action: every click and field is
still a Jev choice among observed ids.

Without a model, simple patterns take over: sites named without ".com" ("go to facebook"), "<item> for 40 euros",
quoted text, URLs and the phrase after "search for". Passwords are never read from a request; they come only from
the Keychain.

### Driving JevOSX from the Claude app (no API key)

Talk to Claude in the Claude app on your Mac, the way you chat with it anyway, and let it drive the Mac through
JevOSX. It uses your Claude plan; no API key is involved.

```bash
jevosx mcp --install
```

This adds JevOSX to the Claude desktop app's connectors (`~/Library/Application Support/Claude/
claude_desktop_config.json`, keeping a `.bak` of the old file) and prints the one command that adds it to Claude Code.
Quit and reopen Claude, then ask in a chat, for example "use JevOSX to open the Marketplace create-listing page and
fill in the title and price". Claude gets four tools:

- `run_task`: one concrete task with the exact text for each field. It runs in the JevOSX console (started for you if
  it is not open), where you see every step and answer its questions and approvals. Claude cannot approve anything.
- `wait_for_run`: keeps following a task that was still going (for example while it waits for you).
- `look_at_screen`: the element table and text of the window JevOSX works in.
- `recent_runs`: the last runs and how they ended.

Jev still makes every click and keystroke behind the confidence gate and the safety policy. `jevosx doctor` shows
whether the connector is installed.

### Claude in the console (API key)

With an Anthropic API key the console can talk to Claude itself (the switch appears once the key is set). Switch the
composer from **Jev** to **Claude** and talk (or type) to Claude instead of giving JevOSX commands. Claude
works out what you want, asks for what only you know, and hands JevOSX one concrete task at a time ("open
facebook.com/marketplace/create/item in Chrome", "type the price 40 into the Price field"). Each task appears as a
normal run card, with its approvals and questions for you. Afterwards Claude reads how the run ended, looks at the
screen when it needs to, and tries a different route when a run did not get there. With voice on, Claude's replies
are spoken, and when it asks you something it listens for your answer.

- Claude's tools are `run_task` (one JevOSX run, waited for, with the exact text for each field it should type),
  `look_at_screen` (the element table and text of the window JevOSX works in) and `recent_runs`, plus web search to
  look up what a site or task needs.
- Jev still makes every click and keystroke, behind the confidence gate and the safety policy: Claude cannot click
  anything itself, and consequential steps still ask you.
- Setup: add `ANTHROPIC_API_KEY=...` to `~/JevOSX/.env` and restart `jevosx ui` (`jevosx doctor` shows whether it is
  set). The `[pilot]` section picks the model (`claude-opus-5-5`), effort, web search and a cap on tool calls per
  message. Requests use the server-side refusal fallback (a declined request is retried on a fallback model).
- The demo console (`jevosx ui --demo`) has a simulated Claude that hands your message to JevOSX as one task.

### Talking it through (questions and voice)

**It asks what only you know, before it starts.** When the on-device model reads your request it also lists what the
task cannot be finished without and only you can give: photos to upload, an item's condition, which account (at most
three questions, never passwords). The console shows them in a *Before I start* card; answer what you can, leave the
rest empty, and press Start. Your answers become text the agent can type. Things it can decide itself (a category, a
title, the wording) it does not ask about. In a terminal the questions are asked in the terminal. Turn it off with
`agent.ask_first = false`.

**A "Your turn" card takes an answer in words.** When Jev needs information mid-run, type it into the card (or do the
step on the Mac) and press *Done, continue*. What you typed becomes text the agent can use.

**Voice.** Press the speaker button at the top of the console to turn voice on:

- JevOSX says its questions, approvals ("Should I click Publish? Say yes or no."), hand-offs and results out loud, with
  the Mac's own voice (`say`). To make it a Siri voice, choose one as the system voice in System Settings ›
  Accessibility › Spoken Content.
- It listens for your answer right after asking: "yes" or "no" for approvals, "done" or the answer for hand-offs,
  an answer or "skip" for each question before starting.
- The microphone button next to Run takes a spoken command (with voice on, it starts right away).

Listening uses the browser's speech recognition, so use Chrome or Safari and allow the microphone for the console
page. Chrome sends the audio to Google's speech service; Safari uses Apple's.

**"Hey Siri, Ask JevOSX".** Siri cannot hold the conversation itself, but it can hand JevOSX the task:

1. Open Shortcuts, create a shortcut named **Ask JevOSX**.
2. Add **Dictate Text**, then **Run Shell Script** (shell `zsh`, pass input *as arguments*) with:
   `$HOME/JevOSX/.venv/bin/jevosx ask "$@"`
3. In Shortcuts › Settings › Advanced, allow running scripts.

With `jevosx ui` open and voice on, "Hey Siri, Ask JevOSX", then "sell my welding gun on Marketplace for 40 euros",
starts the run in the console, which then talks it through with you.

### Logging in to websites

```bash
jevosx login add github.com              # asks for the username and the password (hidden input)
jevosx run "log in to github.com"        # you approve before the password is typed
jevosx login list                        # sites and usernames; passwords are never shown
jevosx login remove github.com
```

Passwords go into the macOS login Keychain (service `jevosx:github.com`). `~/.jevosx/logins.json` only lists sites
and usernames. During a run a saved login becomes two text slots, `login_username` and `login_password`, which
exist only while the page on screen is that site over https:

- The page address comes from the browser's accessibility tree (`AXURL`), not the address bar.
- `github.com` also covers `www.github.com` and other subdomains, but never a look-alike such as
  `github.com.evil.io`. Shared-hosting domains (`github.io`, `vercel.app`, …) match their exact host only.
- Save the login under the site of the sign-in page: `google.com` covers `accounts.google.com`.

Jev sees `•••••• (saved password for github.com)`, never the password. The password may only go into a password
field; you approve it first (`safety.confirm_credentials`); and the executor re-checks the field's own page URL just
before typing.

For what only you can do (a two-factor code, a CAPTCHA, a passkey or Touch ID prompt), Jev picks `ASK_USER`.
In the terminal the run pauses until you press Enter. In the console a **Your turn** card appears with
**Done, continue** and **Stop the run**. After a handoff, Jev looks at the screen again and carries on.

### Apps that draw their own interface

Games, canvas design tools and remote desktops paint pixels instead of exposing Accessibility controls. When the
focused window gives Accessibility almost nothing (fewer than `observer.vision_min_controls` controls and hardly
any text), JevOSX:

1. captures just that window (`screencapture -l`, deleted right after);
2. reads its text with Apple's on-device Vision OCR;
3. offers each line as an element such as `[12] on-screen text "New Game"`, plus a `keyboard` element that types
   at the cursor.

Jev chooses ids as usual. A click lands on the centre of the recognized text, computed from the window's frame.
The model never sees or outputs coordinates. An unchanged window reuses the previous OCR result.

This needs the **Screen Recording** permission for your terminal (`jevosx doctor` asks for it). Without it,
everything else keeps working. `observer.vision = "always"` also reads rich apps (slower, noisier), and `"off"`
disables it. `jevosx observe` shows a `vision:` line whenever OCR was used.

### Python API

```python
from jevosx import Agent, Settings

settings = Settings.load()  # jevosx.toml / ~/.config/jevosx/config.toml + .env + env
with Agent.from_settings(settings) as agent:
    for event in agent.iter_run(
        'Create a new document and type "hello"',
        app="TextEdit",
        verifier=lambda obs: "hello" in obs.text,  # independent success check for DONE
    ):
        print(event.step, event.status, event.action, event.decision["gate_confidence"], event.timings)
    result = agent.last_result
    print(result.status, result.steps, result.elapsed_ms)
```

See [`examples/run_agent.py`](examples/run_agent.py) for a complete script with verification, confirmation and
feedback, and [`examples/dump_tree.py`](examples/dump_tree.py) for inspecting observations.

The layers are independent and easy to swap. `Agent` takes any `Observer` and `Executor` (see
`jevosx/observer/base.py` and `jevosx/executor/base.py`), a `JevRouter`, an optional `MemoryStore`, and hooks for
`confirm` and `on_low_confidence`.

## Memory and the learning loop

- **Storage.** `~/.jevosx/memory.db` (SQLite, WAL) holds `episodes` (goal, status, model, timings) and `steps`
  (app, window, a compact screen summary, operation, stable target key, probability, confidence, outcome).
  Goal and screen embeddings are stored with the rows. Nothing is uploaded.
- **Stable target keys.** Elements are remembered by role + AX identifier or normalized label + container, with
  counts normalized so `Inbox (3)` matches `Inbox (12)`. Menu commands, keys, apps and windows use their own keys.
- **Embeddings.** Deterministic signed feature hashing over words, bigrams and character trigrams (numpy). There
  is no model download, it works offline, and it is stable across processes.
- **Retrieval.** Top goal-similar finished episodes → steps taken on similar screens → mapped onto the current
  action space. Only steps whose target is on screen now become hints:
  - `worked`: changed the UI in a successful or done run;
  - `no_effect`: changed nothing;
  - `finished`: this screen was the final state of a successful run, which is evidence for `DONE`.

  Hints are added to `state.memory_hints` and annotated directly on the matching target option. Jev's instructions
  describe them as weak evidence.
- **Feedback.** A run verified by `--expect-text` / `verifier` is stored as `success` (weight 1.0). A plain
  `DONE` is stored as `done` (0.6). `--feedback`, `agent.feedback(id, ok)` or `jevosx memory label ID success|failed`
  promote or demote a run. Failed and blocked runs only contribute `no_effect` hints.
- **Maintenance.**
  ```bash
  jevosx memory stats | list | show ID | label ID success | forget ID | prune --keep 2000
  jevosx memory export trajectories.jsonl      # portable JSON-lines (no vectors; rebuilt on import)
  jevosx memory import trajectories.jsonl
  ```
  Typed text is not stored unless `memory.store_typed_text = true`. Secret slots are never stored.

## Configuration

Copy [`config/jevosx.example.toml`](config/jevosx.example.toml) to `./jevosx.toml` or
`~/.config/jevosx/config.toml`, or pass `--config PATH`. It lists every setting with its default. Unknown keys
and wrong types are rejected when the config loads.

| Environment variable | Purpose |
| --- | --- |
| `TYPESAFE_API_KEY` | Jev API key (required for `run`) |
| `JEV_MODEL` | Jev model, default `jev-latest` (pin e.g. `jev-1.13.0` for reproducibility) |
| `TYPESAFE_ENDPOINT` | override the System One endpoint |
| `JEVOSX_CONFIG`, `JEVOSX_MEMORY_PATH`, `JEVOSX_MAX_STEPS` | config file, memory database, step budget |
| `TEXT_MODEL`, `TEXT_MODEL_BASE_URL`, `TEXT_MODEL_API_KEY` | optional OpenAI-compatible writer for `TYPE_TEXT` |

A `.env` file in the working directory is loaded automatically. Real environment variables always win.

**Where `TYPE_TEXT` values come from.** Jev chooses; it never writes text. Values come from:
1. the goal itself:
   - quoted literals (`"…"`, `“…”`, `` `…` ``) become slots `quote_1`, …;
   - the phrase after *search for / look for / look up / google / type / enter / say* becomes `phrase_1`, …
     ("look for pictures of red flowers in Safari" → `pictures of red flowers`);
   - URLs and domains become `url_1`, …;
2. `--slot NAME=TEXT` / `text_slots={...}`. With several slots, Jev picks the right one in the same request;
3. optionally, a small OpenAI-compatible model (`[text_model]`, off by default), offered as `GENERATE`.

If none of these is available, `TYPE_TEXT` is not offered at all.

## Performance

- **Jev.** TypeSafe reports that Jev answers typed questions in one parallel pass in roughly 70–500 ms, with free
  output tokens. JevOSX makes **one request per decision**: operation and every target head share one round trip
  on a pooled HTTP/2 connection, so the TLS handshake is paid once. `jevosx doctor --live` measures your actual
  round trip.
- **Observation.** This is usually the larger cost. It scales with the size of the frontmost window's tree:
  small native windows take tens of milliseconds, large web pages take longer. Limits: `observer.time_budget_s`
  (1.5 s), `max_nodes`, `max_elements`. Menu walks are cached per app for `menu_cache_ttl_s`. `jevosx observe`
  prints nodes visited and walk time.
- **Settling.** After an action the loop polls a cheap signature (frontmost app, window title, focused element)
  every 50 ms until it is stable, capped at 0.8 s, instead of sleeping for a fixed time.
- Per-step timings (`observe`, `recall`, `decide`, `jev`, `act`, `settle`) are printed by `jevosx run` and included
  in every `StepEvent` and trace line.

## Limitations

- Apps that draw their own interface are read through OCR, so only controls with visible text can be clicked.
  Icon-only buttons, drag-and-drop, drawing and real-time games are out of reach for now (see the roadmap). The
  opt-in `executor.pointer_fallback` clicks an element's own AX frame centre, but it still needs the element to
  exist.
- The observer walks one window, the agent's work window (including attached sheets), plus open menus. Other
  windows are reachable through `FOCUS_WINDOW`. A window that an action opens in another, already running app is
  not followed automatically; the agent switches to that app with `OPEN_APP`.
- Secure input (password prompts, some banking apps) can block synthetic keystrokes system-wide.
- Some cross-app menus are populated lazily and can be stale for up to `menu_cache_ttl_s`. Pressing a menu item
  that has become disabled fails safely and shows up as a failed step.
- `DONE` is Jev's judgement. Use `--expect-text` or a `verifier` when outcomes matter.
- Platform-independent logic (walker, menus, router, client, memory, safety, agent loop) is unit-tested on every
  OS. The pyobjc-backed modules (`observer/ax.py`, `observer/desktop.py`, `executor/mac.py`,
  `executor/input.py`) are covered in CI by macOS symbol and smoke tests. Driving real apps needs an interactive
  macOS session with the Accessibility permission granted.

## Development

```bash
pip install -r requirements-dev.txt -e .
pytest -q            # offline: fake AX trees, a fake desktop state machine, a mocked Jev endpoint
ruff check . && ruff format --check .
mypy jevosx
```

CI (`.github/workflows/ci.yml`) runs lint, type checks and tests on Ubuntu (Python 3.11 and 3.12) and on macOS.
On macOS the real pyobjc bridges are installed and every Accessibility and CGEvent symbol the code uses is checked.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Accessibility access is not granted` | Enable your terminal/IDE in System Settings › Privacy & Security › Accessibility, then restart it. After a Python upgrade, remove and re-add the entry. |
| Browser pages show almost no elements | Give Chrome/Electron a second after launch (web accessibility turns on lazily). In Safari, make sure the page has finished loading. |
| Runs end with `low_confidence` | The screen is ambiguous for the goal. Make the goal more specific, add `--app`, use `--on-low-confidence ask`, or lower `--min-confidence`. Review `~/.jevosx/fallbacks.jsonl`. |
| `TYPE_TEXT` never happens | Put the text in quotes in the goal or pass `--slot`. For new text ("write a poem"), run `jevosx write --check`. |
| `jevosx write --check` says Apple Intelligence is off | System Settings › Apple Intelligence & Siri → turn it on; the model downloads in the background. `sdkMissing` means the Command Line Tools are older than macOS 26: update them, then `jevosx write --rebuild`. |
| A game or canvas app shows nothing to click | Grant Screen Recording to your terminal (System Settings › Privacy & Security › Screen & System Audio Recording), restart it, and check `jevosx observe --delay 3` for a `vision:` line. |
| Keystrokes go to the wrong app | The executor re-activates the observed app and re-validates the frontmost pid before input. Avoid switching apps during a run. |
| `Jev rejected the API key` | Check `TYPESAFE_API_KEY` (`jevosx doctor`). |
