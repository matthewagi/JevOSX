# Roadmap: from "clicks what it can read" to "any app, any task"

JevOSX 0.1 works well where macOS apps expose an Accessibility tree (TextEdit, Finder, Safari, Chrome). Three gaps
stopped it from handling arbitrary tasks:

| Gap | Example that failed | Why |
| --- | --- | --- |
| Free-form writing | "write a poem about autumn in TextEdit" | Jev picks options; it never writes text. Only quoted text could be typed. |
| Website logins | "log in to github.com" | No safe place to keep passwords, and no way to know which site is on screen. |
| Custom-drawn apps | games, Blender, canvas design tools | They draw pixels instead of Accessibility controls, so the element table is empty. |

One rule shapes every phase: **the decision model only ever chooses among ids that JevOSX produced locally.**
Each new ability is therefore a new local *source of text* or a new local *source of options*. Nothing lets a
model output coordinates, commands or passwords.

## Phase 1: free-form text, written on this Mac (done)

**Research.**
- macOS 26 ships a ~3B-parameter language model that runs on-device: Apple's Foundation Models framework, needing
  Apple Intelligence and Apple silicon. It is free, private and works offline.
- Apple's Python SDK (`apple-fm-sdk` 0.2) is source-only, and its build insists on the full Xcode 26. Its Swift
  bindings show the whole API surface we need:
  - `SystemLanguageModel.default.availability`
  - `LanguageModelSession(model:tools:instructions:)`
  - `GenerationOptions.temperature` and `GenerationOptions.maximumResponseTokens`
  - `session.respond(to:options:)`
- The `fm` command-line tool only ships with macOS 27.

**Design.**
- `jevosx/writer/apple_writer.swift` is about 150 lines of Swift. JevOSX compiles it once with the Command Line
  Tools (`xcrun swiftc`) into `~/.jevosx/bin/`.
  - Protocol: one JSON request on stdin, one JSON answer on stdout.
  - `--check` reports whether the model is available and, if not, why (Apple Intelligence off, device not
    eligible, model still downloading, SDK too old).
  - `#if canImport(FoundationModels)` keeps it compiling on older SDKs, where it reports `sdkMissing` instead of
    breaking the install.
- The writer only runs when Jev picks the `GENERATE` text option for a field Jev chose. Jev still decides *where*
  and *whether* to type; the local model decides *what*.
- `GENERATE` is offered only when the goal asks for composed text (write, draft, reply, summarize, poem, …), so
  goals that worked before are not given new options.
- Guardrails:
  - Never writes into password fields.
  - Screen text is passed to the writer as data.
  - The output is cleaned: no preambles, quotes or code fences.
  - Text is generated once per field per run and reused on retries, so a retry doesn't write a different poem.
- Backends: `writer.backend = auto | apple | openai | off`. `auto` prefers the on-device model. The optional
  OpenAI-compatible text model remains as the fallback for Intel Macs or older macOS.
- Tools:
  - `jevosx write "a haiku about the sea"` to try it.
  - `jevosx write --check` to diagnose it.
  - `jevosx doctor` shows its state.
  - The demo console has a simulated writer.

## Phase 2: logging in to websites (done)

**Research.**
- Passwords saved by Safari or Chrome can't be read by other apps (iCloud Keychain and Chrome's own store), and
  that's the right boundary. JevOSX keeps its own entries in the login Keychain through
  [`keyring`](https://pypi.org/project/keyring/), under the service `jevosx:<host>`. A small index,
  `~/.jevosx/logins.json`, lists hosts and usernames and never stores passwords.
- Safari and Chrome expose the page address as `AXURL` on the `AXWebArea` element. That is enough to bind a
  credential to a site without reading the address bar.

**Design.**
- `jevosx login add github.com` prompts for the password with hidden input. `jevosx login list` and
  `jevosx login remove` manage the entries.
- A saved login becomes two text slots, `login_username` and `login_password`. They exist only while the page on
  screen is https and its host is the saved host or one of its subdomains, so `github.com.evil.io` never matches.
  The password slot is masked for Jev (`••••••`).
- The password may only be typed into a secure (password) field. The executor re-reads the field's page URL
  immediately before typing and refuses if the host changed, which protects against phishing and redirects.
- Every time a saved password is about to be typed, you approve it first (`safety.confirm_credentials`, on by
  default).
- Passwords never reach Jev, the writer, memory, logs or the run history. History shows `(saved password)`.
- A new operation, `ASK_USER`, hands control to the human for what only they can do: 2FA codes, CAPTCHAs,
  passkeys and Touch ID. The reason is a typed choice (2fa · captcha · passkey · missing info · permission · other),
  and the run resumes after the user clicks Continue.

## Phase 3: apps that draw their own interface (done)

**Research.**
- The `screencapture -x -o -l <window id>` command captures one window. It needs the Screen Recording permission,
  which `CGPreflightScreenCaptureAccess()` checks without prompting.
- Apple's Vision framework (`VNRecognizeTextRequest`, accurate level) returns text lines with normalized bounding
  boxes, on-device, in a few hundred milliseconds.

**Design (still coordinate-free for the model).**
- When the Accessibility tree of the focused window is sparse (`observer.vision = "auto"`), JevOSX captures that
  window and runs OCR:
  - Each recognized text becomes an element such as `[42] on-screen text "Play"` with operation `CLICK`.
  - Its click point is the centre of the OCR box, computed locally from the window's frame. Jev picks `42` and
    never sees a coordinate.
- Recognized lines are also added to `visible_text`, so Jev can tell when "Level complete" appears.
- A `keyboard` pseudo-target types at the current cursor. It is offered only in this mode and never selects all
  first, since in a canvas app Cmd-A would select every object.
- OCR text that merely repeats an Accessibility label at the same place is dropped.
- Nothing runs without the Screen Recording permission. The observer then notes "vision unavailable" and
  continues with the Accessibility tree alone.
- `observer.vision = "always"` also OCRs rich apps (noisy, slower). `off` disables the feature.

## Phase 4: planning and handoff (done)

- **Reading the request with the on-device model.** Once per run the local model reads the request and returns
  its steps and the exact values to type: website, title, price, search words, and a written description when
  asked. The values become text slots and the steps a plan, both hints for Jev, never actions.
  - Patterns cover the case without a model: "go to facebook", "<item> for 40 euros", quotes, URLs.
  - Requests are messy ("prepare a product to sell on marketplace a plastic welding gun for 40 euros generic text"),
    so regexes alone were not enough.
  - `agent.plan = auto | off` controls it.
- **Same-page links.** Search results link the same page several times, which split Jev's confidence below the
  floor (seen live: "look up the population of Malta on Wikipedia"). Links now carry their destination, which Jev
  sees as `to`. Probability spread over links to the chosen page counts as one choice.
- **Handoff.** `ASK_USER` (see Phase 2) works for any app, not only logins.

## Phase 5: working behind your window (done)

Seen live: approving a step in the console brought the console window to the front, and the approved typing then
went to the console instead of the new Facebook window. The run failed and kept switching windows.

- **A work window of its own.** The agent keeps the window its own actions brought forward and reads it where it
  is (Accessibility needs no focus). A window the person switches to is never adopted, and neither is the console.
- **Keys only where they belong.** Key presses, menu commands and browser typing bring the work window forward and
  check that it is the key window before anything is sent; otherwise nothing is sent. Afterwards the person's
  window comes back. `agent.background = true | false`.
- **One destination, one choice.** The model's "facebook.com/marketplace" and the pattern's
  "facebook.com/marketplace/create/item" were offered side by side and split Jev's text choice (0.48). Only the more
  specific address is kept now. Chrome also lists its toolbar twice, which gave two identical address bars. Controls
  with the same role, label and frame are offered once.
- **Research: a desktop of its own.** macOS has no public API for a hidden Space or a virtual display
  (`CGVirtualDisplay` is private), so typing still shows the work window for a moment. Candidates: per-process key
  events (`CGEventPostToPid`, unreliable in Chromium), and AppleScript navigation for browsers (`set URL of active
  tab`, which needs the Automation permission).

## Phase 6: confidence that fits the step (done)

Seen live: one 0.65 floor for every step meant the console asked about opening a browser window, focusing the
address bar and typing "facebook.com" (0.43 to 0.63), as often as about publishing. Jev's confidence on a routine
step is often split between equally good routes (Cmd-L, or a click on the address bar), which is not doubt about
the outcome.

- **Risk tiers** (`jevosx/risk.py`). Safe steps (easily undone) need 0.35, routine clicks and Return 0.5, and
  consequential steps (the safety policy's confirm list, close, quit) keep 0.65 and are still confirmed.
- **Remembered steps.** A step that matches one that worked in a similar earlier run counts as safe, so approving
  it once teaches the agent. Careful steps are always asked about.
- **One question per step.** Approving an unsure consequential click also answers the safety confirmation.
- **Text that fits the field.** Seen live: "plastic welding gun" went into Chrome's address bar and became a Google
  search. The text question was answered before Jev knew which field the text was for. Now only fitting text is
  considered for the chosen field (a browser's address bar takes addresses and search words; page fields take
  anything but addresses, unless they ask for one), Jev's answer is renormalized over it, and when it is still
  unsure Jev is asked again with the field in view.
- **Addresses are opened.** Typing an address or search words into a browser's address bar presses Return.
- **One field, one intent.** Jev's probability split between clicking a field and typing into it counts as one.
- **One address bar.** Chrome can list a second copy of its address bar with another value; only one is offered.
- **Look again before asking.** With "ask", an unsure step is re-observed once before you are asked
  (`agent.ask_after_retries`); pages that are still loading made many of the questions.
- **Values the request implies.** Seen live: the listing stopped at Category, which the request never names. The
  on-device model now also gives the everyday category of an item to sell ("category: Tools"), so it can be
  typed or chosen from the list. Facts only the person knows (condition, age, size) are never invented: the
  agent hands those to you (`ASK_USER`, "provide information the goal does not include").
- Next: calibrate the tiers from the fallback log (how often an approved step was right) instead of fixed numbers.

## Phase 7: talking it through (done)

Asked for: only steps with consequences should need confidence or approval (publishing, sending, failed logins);
the agent should find out what it is missing before it starts and ask only for the essentials; and it should talk.

- **Consequences decide.** Safe steps need 0.2, routine clicks 0.3; publishing, sending, paying, deleting,
  quitting keep 0.65 and ask. A second sign-in attempt in a run (after a password was typed) always asks, because
  repeated failures can lock the account.
- **Ask first.** The goal reader lists at most three things only the person can give (photos, condition, which
  account). The console asks them in one card before starting; answers become text slots.
- **Answerable hand-offs.** "Your turn" takes a typed or spoken answer, which becomes a text slot.
- **Voice.** The console speaks questions, approvals and results with the Mac's voice (a Siri voice when chosen as
  the system voice) and listens for yes/no, done, or the answer. `jevosx ask` lets a Siri Shortcut start runs.
- **Photos.** `CMD_SHIFT_G` (Go to folder) lets the agent type the path you gave into an Open dialog.
- Next: decide between "look first" and "ask first" per task (open the form, read which fields are required, then
  ask), with a web search when the model does not know what a site needs.

## Phase 8: Claude in the console (done)

Asked for: talk to Claude, which goes in, tries things and sees how JevOSX reacts.

- **Claude as the head, Jev as the hands.** `jevosx/pilot.py` runs a conversation with Claude (Anthropic API,
  `claude-opus-5-5`, streamed, adaptive thinking, prompt caching, server-side refusal fallback). Its tools:
  `run_task` (one JevOSX run, waited for, summarized with the screen afterwards), `look_at_screen`, `recent_runs`,
  and web search. Claude never clicks: every action is still a Jev choice behind the gate and the safety policy.
- **One feed.** Claude's replies stream into chat bubbles; each task it hands over appears as a normal run card
  ("Claude → JevOSX: …") with its approvals and questions in place.
- **Voice.** Claude's replies are spoken; a question from Claude opens the microphone for the answer.
- **Guardrails.** Tool inputs are validated before anything runs, a cut-off or declined reply never runs its tools,
  a cap stops runaway loops (`pilot.max_tool_calls`), and Stop ends both the run and Claude's turn.
- **Exact text.** Claude passes the text for each field with the task (`texts`, e.g. title and price); it becomes
  the run's text slots, so JevOSX never has to cut the words to type out of a sentence. Passwords are refused.
- Next: a screenshot of the work window for Claude, for pages the accessibility tree describes poorly.

## Phase 9: the Claude app drives JevOSX (done)

Asked for: not through the API, in the chat. The Claude app on the Mac gets JevOSX as a connector (`jevosx mcp`, the
Model Context Protocol over stdio), so the conversation is the person's own Claude chat on their plan.

- Tools: `run_task` (through the open console, started if needed, with exact texts), `wait_for_run`,
  `look_at_screen` (the agent's work window, not the console in front) and `recent_runs`.
- Every task is a normal console run ("Claude → JevOSX: …"); questions and approvals stay with the person there.
- `jevosx mcp --install` writes the Claude desktop app's connector entry (with a backup) and prints the Claude Code
  command. The console's own Claude switch only shows once an API key is set.

## Phase 10: saving pictures from the web (done, not yet seen live)

Asked for: test tasks on the web such as "look for photos and save them in a folder", fast, without questions that
are not needed and without detours.

Before, saving one picture took about seven Jev decisions (open it, right-click, Save Image As…, Go to folder, a
folder that may not exist, a file name, Save), there was no right-click at all, and the goal reader could ask where
to save and how many before starting.

- **SAVE_IMAGE** (`jevosx/images.py`). Photo-sized pictures on web pages that have an address (their AXURL) are
  offered as targets; Jev picks one per step and it is downloaded straight into the folder. Never replaces a file,
  only inside your home folder, only pictures (type checked), at most 25 MB.
- **Folder and count come from the request** ("3 photos", "a folder called dogs on my desktop"). When it does not
  say: ~/Pictures/<what they show>, and 5 pictures. Questions about either are dropped before starting.
- **Straight to picture results.** The run gets a Google Images address for the topic as text to type, so the
  browser lands on pictures in one step.
- **Done when enough are saved**, without waiting for Jev to decide DONE. A saved picture is never offered again.
- **Taste is not doubt.** Several equally good pictures split Jev's target probability; that choice is not gated
  (the operation still is). Saving is a safe step.
- Pictures are only offered to runs that save pictures, so other tasks' state does not grow.
- Seen live (first run, 8 steps, nothing saved): pictures were found and SAVE_IMAGE was offered, but the on-device
  reader's plan was "1. open Finder · 2. search: dogs · 3. count: 3 …", Finder was offered because the request says
  "folder", and on Google's picture results Jev followed a link to Unsplash. Now picture goals get a fixed plan
  (browser → picture_search → SAVE_IMAGE), Finder is not offered for them, "name: value" lines are never plan
  steps, and on the picture results for the topic the next picture is saved without asking Jev.
- Seen live (second run): 2 of 3 pictures saved on Unsplash in 3 steps, then Jev clicked a carousel's "scroll list
  to the right" and "left" buttons in turn until out of steps. A page a picture was saved from is now a source like
  the results page: the next picture is saved without asking, and when all visible ones are saved the page is
  scrolled down (up to 4 times in a row) before Jev decides again.
- Seen live (third run): done after 2 of 3, because Jev said DONE (0.36) and only saves were checked against the
  count; and on Google's picture results no picture was offered, because the results are buttons and only link
  cards were opened. DONE is now refused until enough are saved, and big web buttons are opened like links.
- Seen live (fourth run): done in 4 steps, but all three files were Google's thumbnails (500 to 680 pixels, 30 to
  45 KB, from encrypted-tbn0.gstatic.com). SAVE_IMAGE now presses the tile (AXPress, although the image lists only
  AXShowMenu), reads the original from the preview beside the results or from the tile's /imgres?imgurl= link, and
  saves that; when the site refuses, Google's copy is saved and the step says so. Next run: 2560×1707 (534 KB),
  800×1000, 652×515, still 4 steps.
- Seen live (fifth run, from a fresh window): 6 steps. Reads taken while the results loaded, and again while Chrome
  rebuilt them after a tile was pressed, had only the page's header (40 elements instead of about 108), and Jev
  clicked "Search" and "Search by image". A source page with no unsaved picture and nothing to scroll is now read
  again (up to 10 times) instead of being judged.
- To check live: Chrome's pictures report their address (AXURL). If `jevosx observe` on a Google Images page shows
  no `image` elements, that is the first thing to fix.

## Next research (not built yet)

- **Icons without text.** OCR can't name a play-triangle button. Candidates:
  - Vision saliency (`VNGenerateObjectnessBasedSaliencyImageRequest`) to find clickable blobs, named by position
    and colour ("red round button, top right").
  - Apple's on-device image classification of crops.
- **Drag, draw and real-time control.** Drag-and-drop between two observed ids is a natural next operation. Games
  that need reflexes need a faster local loop (sub-100 ms) than one Jev call per step. Keep them out of scope until
  a local reflex policy exists.
- **ScreenCaptureKit** instead of the `screencapture` CLI (faster, no temp files). It needs a Swift helper like
  the writer's.
- **Native app logins** (for example Slack's desktop sign-in). There is no URL to bind a credential to, so it would
  need an app-bundle binding plus explicit approval.
- **Learning from memory.** Promote episodes that repeatedly succeed into "macros" that Jev can pick as one option.

## Risks and how they are handled

| Risk | Mitigation |
| --- | --- |
| Apple Intelligence off, unsupported language, or model still downloading | `--check` reports the exact reason; `auto` falls back to the OpenAI-compatible model or to no writer. |
| On-device model refuses (guardrails) | The error is surfaced as "writer declined"; the step fails and Jev picks something else. |
| Wrong text for a field | Jev still chooses the field, and the writer never writes into password fields. The text is typed only after the confidence gate and safety checks. |
| Password typed on the wrong site | https plus host binding, a secure-field-only rule, a live URL re-check before typing, and your approval every time. |
| OCR misreads or clicks the wrong spot | OCR elements only appear when Accessibility has too little to offer; the confidence gate still applies; clicks land on the centre of a recognized text line. |
| Screen Recording friction | Asked only when a custom-drawn app is actually in front; everything else keeps working without it. |
