#!/usr/bin/env bash
# JevOSX one-line installer for macOS.
#
#   curl -fsSL https://raw.githubusercontent.com/matthewagi/JevOSX/main/scripts/install.sh | bash
#
# What it does (nothing needs sudo):
#   1. finds Python 3.11+ (or gets one through uv, a single self-contained binary, if none is installed)
#   2. downloads JevOSX to ~/JevOSX (git clone if git works, otherwise a tarball) or updates an existing copy
#   3. creates ~/JevOSX/.venv and installs the dependencies (including the pyobjc macOS bridges)
#   4. asks for your TypeSafe API key (hidden input) and stores it in ~/JevOSX/.env (chmod 600)
#   5. triggers the macOS Accessibility prompt and opens the right System Settings pane
#   6. adds a `jevosx` command and an optional double-clickable "JevOSX.command" launcher on your Desktop
#   7. offers to start the console
#
# Environment overrides: JEVOSX_DIR (install location), JEVOSX_REF (branch), JEVOSX_YES=1 (accept defaults),
# JEVOSX_PYTHON (a python3.11+ path, or "uv" to use a uv-managed Python).
set -euo pipefail

REPO="matthewagi/JevOSX"
DIR="${JEVOSX_DIR:-$HOME/JevOSX}"
FALLBACK_REF="claude/macos-automation-agent-l5e3q2"
YES="${JEVOSX_YES:-0}"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
step() { printf '\n\033[1;34m▸ %s\033[0m\n' "$*"; }
ok() { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die() { printf '\n\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# Prompts read from the terminal even though this script arrives on stdin through `curl | bash`.
TTY=/dev/tty
if ! { true <"$TTY"; } 2>/dev/null; then TTY=""; fi
ask() { # ask "question" default(y|n) → returns 0 for yes
  local question=$1 default=$2 reply=""
  if [ "$YES" = "1" ] || [ -z "$TTY" ]; then [ "$default" = "y" ]; return; fi
  if [ "$default" = "y" ]; then printf '  %s [Y/n] ' "$question"; else printf '  %s [y/N] ' "$question"; fi
  read -r reply <"$TTY" || reply=""
  reply=$(printf '%s' "$reply" | tr '[:upper:]' '[:lower:]')
  if [ -z "$reply" ]; then [ "$default" = "y" ]; else [ "${reply:0:1}" = "y" ]; fi
}

bold "JevOSX installer"
if [ "$(uname -s)" != "Darwin" ] && [ "${JEVOSX_INSTALL_ANY_OS:-0}" != "1" ]; then
  die "JevOSX drives macOS apps, so it installs on a Mac. (Try the simulated console anywhere: jevosx ui --demo.)"
fi

# ---- 1. Python -----------------------------------------------------------------------------------------------------
step "Finding Python 3.11 or newer"
PYTHON=""
CANDIDATES="python3.13 python3.12 python3.11 python3 /opt/homebrew/bin/python3 /usr/local/bin/python3"
case "${JEVOSX_PYTHON:-}" in
  "") ;;
  uv) CANDIDATES="" ;;  # always use a uv-managed Python 3.12
  *) CANDIDATES="$JEVOSX_PYTHON" ;;
esac
for candidate in $CANDIDATES; do
  if command -v "$candidate" >/dev/null 2>&1 &&
    "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
    PYTHON=$(command -v "$candidate")
    break
  fi
done
UV=""
if [ -n "$PYTHON" ]; then
  ok "using $PYTHON ($("$PYTHON" --version 2>&1))"
else
  warn "no Python 3.11+ found (macOS ships 3.9)"
  UV=$(command -v uv || true)
  if [ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ]; then UV="$HOME/.local/bin/uv"; fi
  if [ -z "$UV" ]; then
    ask "Install uv (a small standalone tool from astral.sh) to provide Python 3.12?" y ||
      die "Install Python 3.11+ (https://www.python.org/downloads/ or 'brew install python@3.12') and run this again."
    if ! curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh >/dev/null; then
      die "Could not download uv. Install Python 3.11+ (https://www.python.org/downloads/) and run this again."
    fi
    UV="$HOME/.local/bin/uv"
    [ -x "$UV" ] || die "uv installation failed"
  fi
  ok "using uv to provide Python 3.12 ($UV)"
fi

# ---- 2. Source -----------------------------------------------------------------------------------------------------
step "Downloading JevOSX to $DIR"
REF="${JEVOSX_REF:-}"
if [ -z "$REF" ]; then
  REF="main"
  # Until the pull request is merged, main does not contain the package yet: use the development branch.
  if ! curl -fsSL -o /dev/null "https://raw.githubusercontent.com/$REPO/refs/heads/main/pyproject.toml" 2>/dev/null; then
    REF="$FALLBACK_REF"
  fi
fi
has_git() { command -v git >/dev/null 2>&1 && git --version >/dev/null 2>&1; } # git may be the CLT install stub
if [ -d "$DIR/.git" ] && has_git; then
  git -C "$DIR" fetch --quiet origin "$REF"
  git -C "$DIR" checkout --quiet -B "$REF" "origin/$REF"
  ok "updated existing checkout ($REF)"
elif [ -d "$DIR" ] && [ -n "$(ls -A "$DIR" 2>/dev/null)" ] && [ ! -f "$DIR/pyproject.toml" ]; then
  die "$DIR exists and is not a JevOSX folder. Move it away or set JEVOSX_DIR=/another/path."
elif has_git && [ ! -d "$DIR" ]; then
  git clone --quiet --branch "$REF" "https://github.com/$REPO.git" "$DIR"
  ok "cloned ($REF)"
else
  mkdir -p "$DIR"
  curl -fsSL "https://codeload.github.com/$REPO/tar.gz/refs/heads/$REF" | tar -xz -C "$DIR" --strip-components 1
  ok "downloaded $REF (tarball; .env and .venv are kept on updates)"
fi
cd "$DIR"

# ---- 3. Dependencies -----------------------------------------------------------------------------------------------
step "Installing dependencies into $DIR/.venv (a minute or two the first time)"
if [ -n "$UV" ]; then
  "$UV" venv --quiet --python 3.12 .venv
  "$UV" pip install --quiet --python .venv/bin/python -r requirements.txt -e .
else
  if [ ! -x .venv/bin/python ]; then "$PYTHON" -m venv .venv; fi
  .venv/bin/python -m pip install --quiet --upgrade pip
  .venv/bin/python -m pip install --quiet -r requirements.txt -e .
fi
ok "installed $(.venv/bin/jevosx --version)"

# ---- 4. API key ----------------------------------------------------------------------------------------------------
step "TypeSafe API key"
touch .env
chmod 600 .env
if grep -Eq '^(export )?TYPESAFE_API_KEY=.+' .env; then
  ok "already set in $DIR/.env"
elif [ -n "${TYPESAFE_API_KEY:-}" ]; then
  printf 'TYPESAFE_API_KEY=%s\n' "$TYPESAFE_API_KEY" >>.env
  ok "saved from your environment to $DIR/.env"
elif [ -n "$TTY" ] && [ "$YES" != "1" ]; then
  printf '  Paste your key and press Enter (input is hidden; press Enter alone to skip and use demo mode): '
  KEY=""
  read -rs KEY <"$TTY" || KEY=""
  printf '\n'
  if [ -n "$KEY" ]; then
    printf 'TYPESAFE_API_KEY=%s\n' "$KEY" >>.env
    ok "saved to $DIR/.env (readable only by you)"
  else
    warn "skipped: the console will start in demo mode until you add TYPESAFE_API_KEY=… to $DIR/.env"
  fi
else
  warn "no key given: add TYPESAFE_API_KEY=… to $DIR/.env"
fi

# ---- 5. Accessibility ----------------------------------------------------------------------------------------------
step "Accessibility permission"
case "${TERM_PROGRAM:-}" in
  Apple_Terminal | "") HOST_APP="Terminal" ;;
  iTerm.app) HOST_APP="iTerm" ;;
  vscode) HOST_APP="Visual Studio Code" ;;
  *) HOST_APP="$TERM_PROGRAM" ;;
esac
if [ "$(uname -s)" = "Darwin" ]; then
  if .venv/bin/python -c 'from jevosx.observer.ax import is_trusted; import sys; sys.exit(0 if is_trusted(prompt=True) else 1)'; then
    ok "already granted to $HOST_APP"
  else
    warn "macOS needs your OK before JevOSX can read and control apps."
    echo "    In the window that opens: System Settings › Privacy & Security › Accessibility →"
    echo "    turn ON $HOST_APP (click + and add it if it is not listed). Then quit it with ⌘Q and reopen it."
    open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility" 2>/dev/null || true
  fi
else
  warn "not macOS: skipped"
fi

# ---- 6. Launchers --------------------------------------------------------------------------------------------------
step "Shortcuts"
mkdir -p "$HOME/.local/bin"
ln -sf "$DIR/.venv/bin/jevosx" "$HOME/.local/bin/jevosx"
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ok "command: jevosx" ;;
  *)
    if grep -qs 'Added by the JevOSX installer' "$HOME/.zshrc"; then
      ok "command: jevosx (in new terminal windows)"
    elif ask "Add ~/.local/bin to your PATH (in ~/.zshrc) so 'jevosx' works in new terminals?" y; then
      printf '\n# Added by the JevOSX installer\nexport PATH="$HOME/.local/bin:$PATH"\n' >>"$HOME/.zshrc"
      ok "added; new terminal windows will have the jevosx command"
    else
      ok "command: $HOME/.local/bin/jevosx"
    fi
    ;;
esac
LAUNCHER="$DIR/JevOSX.command"
cat >"$LAUNCHER" <<EOF
#!/bin/bash
# Double-click to open the JevOSX console.
cd "$DIR"
if grep -Eq '^(export )?TYPESAFE_API_KEY=.+' .env 2>/dev/null; then
  exec .venv/bin/jevosx ui
else
  echo "No TypeSafe API key in $DIR/.env yet: starting the simulated demo."
  exec .venv/bin/jevosx ui --demo
fi
EOF
chmod +x "$LAUNCHER"
ok "launcher: $LAUNCHER"
if [ -d "$HOME/Desktop" ] && ask "Put a double-clickable JevOSX launcher on your Desktop?" y; then
  cp "$LAUNCHER" "$HOME/Desktop/JevOSX.command"
  chmod +x "$HOME/Desktop/JevOSX.command"
  ok "Desktop/JevOSX.command"
fi

# ---- 7. Start ------------------------------------------------------------------------------------------------------
echo
bold "JevOSX is installed in $DIR"
echo "  Start the console any time:  jevosx ui        (simulated practice mode: jevosx ui --demo)"
echo "  Check the setup:             jevosx doctor"
if [ -n "$TTY" ] && ask "Start the console now?" y; then
  if grep -Eq '^(export )?TYPESAFE_API_KEY=.+' .env; then
    exec .venv/bin/jevosx ui <"$TTY"
  else
    exec .venv/bin/jevosx ui --demo <"$TTY"
  fi
fi
