"""Example: drive a Mac app toward a goal with the Python API.

    export TYPESAFE_API_KEY=...            # or put it in .env
    python examples/run_agent.py                              # default TextEdit demo
    python examples/run_agent.py --goal 'In Safari, open a new tab and go to "apple.com"' --app Safari

The quoted text in the goal becomes a text slot, so TYPE_TEXT works without any text-generation model.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevosx import Action, Agent, Observation, Settings  # noqa: E402
from jevosx.errors import JevOSXError  # noqa: E402

DEFAULT_GOAL = 'Create a new TextEdit document and type "Hello from JevOSX, driven by accessibility trees."'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--goal", default=DEFAULT_GOAL)
    parser.add_argument("--app", default="TextEdit", help="app to open first ('' to use the frontmost app)")
    parser.add_argument("--expect", default="Hello from JevOSX", help="text that must be visible before DONE counts")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true", help="decide once, execute nothing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    settings = Settings.load()  # ./jevosx.toml, ~/.config/jevosx/config.toml, .env and environment

    def verify(obs: Observation) -> bool:
        """Independent success check: DONE is only accepted when the expected text is on screen."""
        needle = args.expect.lower()
        return needle in obs.text.lower() or any(needle in (e.value or "").lower() for e in obs.elements)

    def confirm(action: Action, reason: str) -> bool:
        return input(f"allow {action.describe()} ({reason})? [y/N] ").strip().lower() == "y"

    try:
        with Agent.from_settings(settings, dry_run=args.dry_run, confirm=confirm) as agent:
            for event in agent.iter_run(
                args.goal,
                app=args.app or None,
                max_steps=1 if args.dry_run else args.max_steps,
                verifier=verify if args.expect else None,
            ):
                decision = event.decision or {}
                timings = event.timings
                print(
                    f"step {event.step:>2}  {event.status:<14} {event.action or ''}"
                    f"  (gate conf {decision.get('gate_confidence', 0):.2f},"
                    f" jev {timings.get('jev_ms', 0):.0f} ms, observe {timings.get('observe_ms', 0):.0f} ms)"
                )
            result = agent.last_result
            assert result is not None
            seconds = result.elapsed_ms / 1000
            print(f"\nresult: {result.status} in {result.steps} steps, {seconds:.1f}s {result.message}")
            # Human feedback promotes the trajectory so future runs retrieve it as a strong hint.
            if (
                result.episode_id is not None
                and result.status == "done"
                and input("did it work? [y/n] ").strip().lower().startswith("y")
            ):
                agent.feedback(result.episode_id, success=True)
            return 0 if result.ok else 1
    except JevOSXError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
