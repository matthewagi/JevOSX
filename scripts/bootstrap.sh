#!/usr/bin/env bash
# Create a virtualenv, install JevOSX with its macOS bridges, and run the environment doctor.
# Usage: scripts/bootstrap.sh [python-executable]
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${1:-python3}"

if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "JevOSX needs Python 3.11+ (found: $("$PYTHON" --version 2>&1)). Try: brew install python@3.12" >&2
  exit 1
fi

"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip >/dev/null
.venv/bin/python -m pip install -r requirements-dev.txt -e .

if [ ! -f .env ]; then
  cp .env.example .env
  echo "created .env: add your TYPESAFE_API_KEY there"
fi

echo
echo "Running checks (macOS will ask to grant Accessibility access to your terminal the first time):"
.venv/bin/jevosx doctor || true
echo
echo "Next:"
echo "  source .venv/bin/activate"
echo "  jevosx observe --delay 3            # see the structured text map of any app"
echo "  jevosx run 'Create a new TextEdit document and type \"hello\"' --app TextEdit --expect-text hello"
