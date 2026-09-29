import json
from datetime import datetime

from jevosx.cli import main
from jevosx.memory import HashingEmbedder, MemoryStore


def test_report_prints_steps_and_withheld_decisions(tmp_path, capsys):
    db, log = tmp_path / "memory.db", tmp_path / "fallbacks.jsonl"
    started = datetime.strptime("2026-09-29T00:39:50+0200", "%Y-%m-%dT%H:%M:%S%z").timestamp()
    store = MemoryStore(db, HashingEmbedder(512))
    reading = {"plan": ["Open a browser window", "Search for red flowers"], "values": {"search": "red flowers"}}
    episode = store.begin_episode("look for pictures of flowers red", meta=reading)
    store._db.execute("UPDATE episodes SET started_at = ? WHERE id = ?", (started, episode))
    store.record_step(
        episode, idx=1, app="com.google.Chrome", window="JevOSX Console - Google Chrome", state_summary="s",
        operation="PRESS_KEY", memory_key="key:CMD_N", target_text="CMD_N (cmd+n)", outcome="changed",
    )  # fmt: skip
    store.finish_episode(episode, "low_confidence")
    store._db.execute("UPDATE episodes SET finished_at = ? WHERE id = ?", (started + 10, episode))
    store.close()
    record = {
        "ts": "2026-09-29T00:39:53+0200",
        "window": "New Tab - Google Chrome",
        "decision": {"operation": "PRESS_KEY", "target": "CMD_L (cmd+l)"},
        "confidence": 0.39,
        "floor": 0.65,
        "risk": "safe step: Focus the address/location bar",
        "resolution": "retry",
        "offered": ["MENU", "DONE"],
        "focused": None,
        "elements": [],
    }
    log.write_text(json.dumps(record) + "\n" + json.dumps({"ts": "2026-01-01T00:00:00+0000"}) + "\nnot json\n")
    config = tmp_path / "jevosx.toml"
    config.write_text(f'[memory]\npath = "{db}"\n[agent]\nfallback_log = "{log}"\n')
    assert main(["--config", str(config), "report"]) == 0
    out = capsys.readouterr().out
    assert "low_confidence" in out and "step 1 PRESS_KEY CMD_N" in out
    assert "withheld 00:39:53 PRESS_KEY CMD_L" in out and "offered: ['MENU', 'DONE']" in out
    assert out.count("withheld") == 1  # the unrelated old record is not attributed to this run
    assert "plan: 1. Open a browser window · 2. Search for red flowers" in out
    assert "to type: search='red flowers'" in out
    assert "safe step: Focus the address/location bar · retry" in out
