"""Saving pictures from the web: "find 3 photos of golden retrievers and save them in a folder called dogs"."""

import base64
from pathlib import Path

import httpx
import pytest

from jevosx.agent import Agent
from jevosx.config import Settings
from jevosx.executor.keys import key_vocabulary
from jevosx.images import ImageSaveError, ImageSaver, image_count, image_task, image_topic, resolve_folder
from jevosx.observer.walker import TreeWalker
from jevosx.planner import GoalReading
from jevosx.router.policy import JevRouter
from jevosx.types import AppInfo, Rect
from tests.fakes import FakeDesktop, FakeNode, element, observation, scripted_client

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
HOME = Path("/Users/ada")
GOAL = "find 3 photos of golden retrievers and save them in a folder called dogs on my desktop"
CHROME = AppInfo("Google Chrome", "com.google.Chrome", pid=300)


# ---- reading the request -------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("goal", "count", "topic", "folder"),
    [
        (GOAL, 3, "golden retrievers", "Desktop/dogs"),
        ("look for photos of red pandas and save them in a folder", 5, "red pandas", "Pictures/Red Pandas"),
        ("download a picture of the Eiffel Tower to my downloads", 1, "Eiffel Tower", "Downloads/Eiffel Tower"),
        ("save ten cat pictures", 10, "cat", "Pictures/Cat"),
        ('save 4 images of sunsets into the "beach" folder', 4, "sunsets", "Pictures/beach"),
        ("grab some wallpapers of mountains and put them in ~/Pictures/walls", 5, "mountains", "Pictures/walls"),
    ],
)
def test_reads_count_topic_and_folder_from_the_request(goal, count, topic, folder):
    task = image_task(goal, home=HOME)
    assert task is not None
    assert (task.count, task.topic, task.folder) == (count, topic, HOME / folder)


def test_only_picture_saving_requests_become_image_tasks():
    assert image_task("open a new Chrome window", home=HOME) is None
    assert image_task("post my photos to facebook marketplace", home=HOME) is None


def test_planner_values_win_and_folders_stay_inside_home():
    task = image_task(GOAL, {"folder": "~/Documents/retrievers", "count": "2"}, home=HOME)
    assert task is not None and task.folder == HOME / "Documents/retrievers" and task.count == 2
    assert resolve_folder("/etc/dogs", HOME) is None
    assert resolve_folder("~/../../etc", HOME) is None
    assert resolve_folder("~", HOME) is None
    assert image_count("save photos of cats", {"count": "500"}) == 50
    assert image_topic("save some nice photos") == ""


# ---- downloading ---------------------------------------------------------------------------------------------------
def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_saves_into_a_new_folder_and_never_replaces_a_file(tmp_path):
    saver = ImageSaver(client(lambda r: httpx.Response(200, content=PNG, headers={"content-type": "image/png"})))
    folder = tmp_path / "Desktop" / "dogs"
    first = saver.save("https://img.example/a.png", folder, "golden retrievers")
    second = saver.save("https://img.example/b", folder, "golden retrievers")
    assert [first.name, second.name] == ["golden retrievers 1.png", "golden retrievers 2.png"]
    assert first.read_bytes() == PNG


def test_data_addresses_and_type_sniffing(tmp_path):
    saver = ImageSaver(client(lambda r: httpx.Response(200, content=PNG)))  # no content type: sniffed
    assert saver.save("https://img.example/x", tmp_path, "x").suffix == ".png"
    data = "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8\xff" + b"\x00" * 8).decode()
    assert saver.save(data, tmp_path, "thumb").suffix == ".jpg"


def test_refuses_what_is_not_a_picture(tmp_path):
    page = ImageSaver(client(lambda r: httpx.Response(200, text="<html>", headers={"content-type": "text/html"})))
    with pytest.raises(ImageSaveError, match="not a picture"):
        page.save("https://img.example/page", tmp_path, "x")
    missing = ImageSaver(client(lambda r: httpx.Response(404)))
    with pytest.raises(ImageSaveError, match="404"):
        missing.save("https://img.example/gone.jpg", tmp_path, "x")
    with pytest.raises(ImageSaveError):
        missing.save("file:///etc/passwd", tmp_path, "x")
    huge = ImageSaver(client(lambda r: httpx.Response(200, content=PNG * 100)), max_bytes=100)
    with pytest.raises(ImageSaveError, match="too large"):
        huge.save("https://img.example/big.png", tmp_path, "x")
    assert list(tmp_path.iterdir()) == []


# ---- what the observer offers --------------------------------------------------------------------------------------
def test_walker_offers_photo_sized_web_pictures_with_an_address():
    web = FakeNode(
        "AXWebArea",
        URL="https://www.google.com/search?q=dogs&udm=2",
        frame=(0, 0, 800, 600),
        children=[
            FakeNode("AXImage", Description="Google", URL="https://www.google.com/logo.png", frame=(10, 10, 90, 30)),
            FakeNode(
                "AXImage",
                Description="Golden retriever puppy",
                URL="https://img.example/1.jpg",
                frame=(10, 60, 180, 140),
            ),  # fmt: skip
            FakeNode(
                "AXLink",
                Title="Golden retriever on grass · example.com",
                URL="https://example.com/dogs",
                frame=(200, 60, 180, 160),
                children=[FakeNode("AXImage", URL="data:image/jpeg;base64,/9j/AAAA", frame=(200, 60, 180, 140))],
            ),
            FakeNode("AXImage", Description="No address", frame=(400, 60, 180, 140)),
        ],
    )
    root = FakeNode("AXWindow", Title="dogs - Google Search", frame=(0, 0, 800, 600), children=[web])
    result = TreeWalker().walk([(root, None)])
    pictures = [e for e in result.elements if e.kind == "image"]
    assert [(p.label, p.url[:22]) for p in pictures] == [
        ("Golden retriever puppy", "https://img.example/1."),
        ("image", "data:image/jpeg;base64"),
    ]
    assert all(p.ops == ("SAVE_IMAGE",) for p in pictures)
    assert any(e.role == "AXLink" and "Golden retriever on grass" in e.label for e in result.elements)


# ---- the whole run -------------------------------------------------------------------------------------------------
def picture(index, label, url):
    return element(
        index, "AXImage", label, ops=("SAVE_IMAGE",), kind="image", url=url, in_web_area=True,
        frame=Rect(0, 0, 180, 140),
    )  # fmt: skip


def screens():
    address_bar = element(1, "AXTextField", "Address and search bar", kind="text_input", ops=("TYPE_TEXT", "CLICK"))
    return {
        "blank": lambda: observation([address_bar], app=CHROME, window="New Tab"),
        "results": lambda: observation(
            [
                address_bar,
                element(2, "AXLink", "Images", in_web_area=True),
                *(picture(i + 3, f"Golden retriever {i}", f"https://img.example/{i}.jpg") for i in range(6)),
            ],
            app=CHROME,
            window="golden retrievers - Google Search",
        ),
    }


def jev(body):
    """A stand-in for Jev on the results page: sure it should save a picture, split evenly over which one."""
    ops = body["questions"]["operation"]["criteria"]
    if "SAVE_IMAGE" in ops:
        head = body["questions"].get("image_target")
        pick = next(iter(head["criteria"])) if head else None
        return {"operation": "SAVE_IMAGE", "image_target": (pick, 0.17)}  # 6 look alike: low target confidence
    return {"operation": "TYPE_TEXT", "text_slot": "picture_search"}


def test_finds_and_saves_pictures_without_asking(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    requests = []
    desktop = FakeDesktop(screens(), "blank", {("blank", "google.com/search"): "results"})
    settings = Settings()
    settings.executor.wait_s = 0
    settings.agent.fallback_log = str(tmp_path / "fallbacks.jsonl")
    settings.agent.low_confidence_policy = "ask"
    asked = []
    downloads = []

    def serve(request):
        downloads.append(str(request.url))
        return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})

    agent = Agent(
        observer=desktop,
        executor=desktop,
        router=JevRouter(scripted_client(jev, requests), keys=key_vocabulary()),
        settings=settings,
        confirm=lambda action, reason: asked.append(reason) or True,
        clarify=lambda questions: asked.append(questions) or {},
        image_saver=ImageSaver(client(serve)),
        sleep=lambda _s: None,
    )
    result = agent.run(GOAL)

    assert result.status == "done" and asked == []
    assert result.steps == 4  # open the picture results, then one step per picture
    assert desktop.executed == [
        'TYPE_TEXT [1] textfield "Address and search bar" <- https://www.google.com/search?q=golden+retrievers&udm=2'
    ]
    folder = tmp_path / "Desktop" / "dogs"
    assert sorted(p.name for p in folder.iterdir()) == [f"golden retrievers {n}.png" for n in (1, 2, 3)]
    assert len(set(downloads)) == 3  # a saved picture is not offered again
    assert "3 of 3" in result.message
    last = requests[-1]["questions"]
    assert len(last["image_target"]["criteria"]) == 4 and "SAVE_IMAGE" in last["operation"]["criteria"]


def test_folder_and_count_questions_are_not_asked(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    class Reader:
        def read(self, goal):
            return GoalReading(
                values={"folder": "~/Desktop/dogs", "count": "3", "search": "golden retrievers"},
                questions=[("folder", "Which folder should I use?"), ("count", "How many photos do you want?")],
            )

    desktop = FakeDesktop(screens(), "blank", {("blank", "google.com/search"): "results"})
    asked = []
    settings = Settings()
    settings.agent.fallback_log = ""
    agent = Agent(
        observer=desktop,
        executor=desktop,
        router=JevRouter(scripted_client(jev), keys=key_vocabulary()),
        settings=settings,
        planner=Reader(),  # type: ignore[arg-type]
        clarify=lambda questions: asked.append(questions) or {},
        image_saver=ImageSaver(client(lambda r: httpx.Response(200, content=PNG))),
        sleep=lambda _s: None,
    )
    result = agent.run("save photos of golden retrievers")
    assert result.status == "done" and asked == []
    assert len(list((tmp_path / "Desktop" / "dogs").iterdir())) == 3


def test_other_runs_do_not_see_pictures(tmp_path):
    requests = []
    desktop = FakeDesktop({"results": screens()["results"]}, "results", {})
    settings = Settings()
    settings.agent.fallback_log = ""
    agent = Agent(
        observer=desktop,
        executor=desktop,
        router=JevRouter(scripted_client(lambda body: {"operation": "DONE"}, requests), keys=key_vocabulary()),
        settings=settings,
        sleep=lambda _s: None,
    )
    assert agent.run("open the Images tab").status == "done"
    state, questions = requests[0]["state"], requests[0]["questions"]
    assert "SAVE_IMAGE" not in questions["operation"]["criteria"] and "image_target" not in questions
    assert [e["label"] for e in state["elements"]] == ["Address and search bar", "Images"]


def test_on_the_picture_results_pictures_are_saved_without_asking_jev(tmp_path, monkeypatch):
    """Seen live: on Google's picture results Jev followed a link to Unsplash instead of saving."""
    monkeypatch.setenv("HOME", str(tmp_path))
    base = screens()

    def results():
        obs = base["results"]()
        obs.page_url = "https://www.google.com/search?q=golden+retrievers&udm=2&sca_esv=abc"
        return obs

    wander = []

    def jev_wanders(body):
        wander.append(body)
        if "SAVE_IMAGE" in body["questions"]["operation"]["criteria"]:
            return {"operation": "CLICK"}
        return {"operation": "TYPE_TEXT", "text_slot": "picture_search"}

    desktop = FakeDesktop({"blank": base["blank"], "results": results}, "blank", {("blank", "google.com"): "results"})
    settings = Settings()
    settings.agent.fallback_log = ""
    agent = Agent(
        observer=desktop,
        executor=desktop,
        router=JevRouter(scripted_client(jev_wanders), keys=key_vocabulary()),
        settings=settings,
        image_saver=ImageSaver(client(lambda r: httpx.Response(200, content=PNG))),
        sleep=lambda _s: None,
    )
    result = agent.run(GOAL)
    assert result.status == "done" and result.steps == 4 and len(wander) == 1  # Jev only opened the results
    assert len(desktop.executed) == 1
    assert agent.last_plan[-1].startswith("SAVE_IMAGE one picture per step until 3")


def test_results_page_and_value_lines_in_the_plan():
    from jevosx.images import is_results_page
    from jevosx.planner import parse_plan
    from jevosx.router.space import intent_apps

    assert is_results_page("https://www.google.com/search?q=Golden+Retrievers&udm=2", "golden retrievers")
    assert not is_results_page("https://www.google.com/search?q=golden+retrievers", "golden retrievers")
    assert not is_results_page("https://www.google.com/search?q=cats&udm=2", "golden retrievers")
    assert not is_results_page("https://unsplash.com/s/photos/golden-retriever?udm=2", "golden retriever")
    seen_live = "1. open Finder\n2. search: dogs\n3. count: 3\n4. move: 3\n5. to: Desktop\n6. folder: dogs"
    assert parse_plan(seen_live) == []  # one real step is no plan
    assert "finder" not in intent_apps(GOAL, set())
    assert "finder" in intent_apps("open my downloads folder", set())
