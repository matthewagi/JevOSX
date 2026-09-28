"""Print what the agent sees: the indexed text map of the frontmost window (no API key needed).

python examples/dump_tree.py --delay 3            # switch to the app you want within 3 seconds
python examples/dump_tree.py --delay 3 --json     # the exact Jev state + choice questions
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevosx import Settings  # noqa: E402
from jevosx.executor.keys import key_vocabulary  # noqa: E402
from jevosx.observer import create_observer, render_text_map  # noqa: E402
from jevosx.observer.ax import require_trusted  # noqa: E402
from jevosx.router.policy import build_state, state_size  # noqa: E402
from jevosx.router.space import ActionSpace  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--menus", action="store_true")
    args = parser.parse_args()

    settings = Settings.load()
    observer = create_observer(settings.observer)
    require_trusted()
    print(f"observing the frontmost app in {args.delay:.0f}s…", file=sys.stderr)
    time.sleep(args.delay)
    started = time.perf_counter()
    obs = observer.observe()
    elapsed = (time.perf_counter() - started) * 1000
    if args.json:
        state = build_state(obs)
        space = ActionSpace.build(obs, keys=key_vocabulary(), text_available=False)
        print(json.dumps({"state": state, "operations": list(space.operations)}, indent=2, ensure_ascii=False))
        print(f"state: {state_size(state)} bytes", file=sys.stderr)
    else:
        print(render_text_map(obs, menus=args.menus))
    print(f"observed in {elapsed:.0f} ms", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
