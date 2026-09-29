import numpy as np
import pytest

from jevosx.config import MemorySettings
from jevosx.executor.keys import key_vocabulary
from jevosx.memory import HashingEmbedder, MemoryRetriever, MemoryStore, state_summary
from jevosx.router.space import ActionSpace
from tests.fakes import element, observation


def test_embedder_is_deterministic_normalized_and_semantic():
    a, b = HashingEmbedder(256), HashingEmbedder(256)
    v = a.embed("Send an email to Bob about lunch")
    assert np.allclose(v, b.embed("Send an email to Bob about lunch"))
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)
    near = float(v @ a.embed("send email to Alice about lunch"))
    far = float(v @ a.embed("resize the photo to 50 percent"))
    assert near > 0.5 > far
    assert np.allclose(a.from_bytes(a.to_bytes(v)), v, atol=1e-3)


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "memory.db", HashingEmbedder(128))
    yield s
    s.close()


def compose_screen():
    return observation(
        [element(1, "AXButton", "Compose"), element(2, "AXButton", "Archive"), element(3, "AXButton", "Refresh")],
        window="Inbox",
    )


def record_run(store, goal, status, steps):
    obs = compose_screen()
    episode = store.begin_episode(goal, app="com.apple.TextEdit")
    for idx, (op, key, outcome) in enumerate(steps, start=1):
        store.record_step(
            episode, idx=idx, app="com.apple.TextEdit", window="Inbox", state_summary=state_summary(obs),
            operation=op, memory_key=key, target_text=key, outcome=outcome,
        )  # fmt: skip
    store.finish_episode(episode, status)
    return episode


def test_store_roundtrip_stats_label_prune_export_import(store, tmp_path):
    compose = "el:" + compose_screen().elements[0].signature
    first = record_run(store, "write an email to bob", "done", [("CLICK", compose, "changed")])
    record_run(store, "archive old mail", "blocked", [("CLICK", "el:x", "unchanged")])
    store.label_episode(first, "success")
    stats = store.stats()
    assert stats["episodes"] == 2 and stats["steps"] == 2 and stats["by_status"]["success"] == 1
    assert store.episode(first).status == "success"
    with pytest.raises(ValueError):
        store.label_episode(first, "done")

    path = tmp_path / "export.jsonl"
    assert store.export_jsonl(path) == 2
    other = MemoryStore(tmp_path / "other.db", HashingEmbedder(128))
    assert other.import_jsonl(path) == 2
    assert other.stats()["steps"] == 2
    other.close()

    assert store.prune(keep=1) == 1
    assert [e.status for e in store.episodes()] == ["success"]


def test_dimension_mismatch_is_refused(tmp_path):
    s = MemoryStore(tmp_path / "m.db", HashingEmbedder(128))
    s.begin_episode("x")
    s.close()
    with pytest.raises(RuntimeError, match="128-d"):
        MemoryStore(tmp_path / "m.db", HashingEmbedder(256))


def test_retriever_turns_similar_successful_steps_into_actionable_hints(store):
    obs = compose_screen()
    compose, archive = ("el:" + e.signature for e in obs.elements[:2])
    record_run(store, "write an email to bob about lunch", "success", [("CLICK", compose, "changed")])
    record_run(
        store, "write an email to carol", "done", [("CLICK", compose, "changed"), ("CLICK", archive, "unchanged")]
    )
    record_run(store, "write an email to dan", "failed", [("CLICK", "el:gone|x", "changed")])
    record_run(store, "defragment the disk", "success", [("CLICK", archive, "changed")])

    space = ActionSpace.build(obs, keys=key_vocabulary(), text_available=False)
    hints = MemoryRetriever(store, MemorySettings(min_goal_similarity=0.3)).hints("write an email to erin", obs, space)
    worked = [h for h in hints if h.kind == "worked"]
    assert worked and worked[0].target_id == "1" and worked[0].support == 2
    assert any(h.kind == "no_effect" and h.target_id == "2" for h in hints)
    assert all(h.target_id in (None, "1", "2") for h in hints)  # unrelated / unresolvable steps never leak in
    assert worked[0].to_state()["past_runs"] == 2 and "2 similar past runs" in worked[0].note()


def test_retriever_is_empty_without_history(store):
    obs = compose_screen()
    space = ActionSpace.build(obs, keys=key_vocabulary(), text_available=False)
    assert MemoryRetriever(store).hints("anything", obs, space) == []
