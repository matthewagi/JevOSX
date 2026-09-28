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

## Phase 2: logging in to websites (next)

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
- The first entry of a password asks for approval (`safety.confirm_credentials`, on by default).
- Passwords never reach Jev, the writer, memory, logs or the run history. History shows `(saved password)`.
- A new operation, `ASK_USER`, hands control to the human for what only they can do: 2FA codes, CAPTCHAs,
  passkeys and Touch ID. The reason is a typed choice (2fa · captcha · passkey · missing info · permission · other),
  and the run resumes after the user clicks Continue.

## Phase 3: apps that draw their own interface (next)

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

## Phase 4: planning and handoff (next)

- **On-device plan.** For multi-part goals ("write a poem, save it as poem.rtf, then open it in Pages") the local
  model splits the goal into ordered steps once per run. The plan is sent to Jev as context: a suggested outline
  from a small model, never a command. Every action is still a Jev choice among observed ids. It is shown in the
  CLI and console, and `agent.plan = auto | off` controls it.
- **Handoff.** `ASK_USER` (see Phase 2) works for any app, not only logins.

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
| Password typed on the wrong site | https plus host binding, a secure-field-only rule, a live URL re-check before typing, and first-use approval. |
| OCR misreads or clicks the wrong spot | OCR elements only appear when Accessibility has too little to offer; the confidence gate still applies; clicks land on the centre of a recognized text line. |
| Screen Recording friction | Asked only when a custom-drawn app is actually in front; everything else keeps working without it. |
