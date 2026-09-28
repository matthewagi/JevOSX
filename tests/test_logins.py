import json
import os

import pytest

from jevosx.agent import Agent
from jevosx.config import SafetySettings, Settings
from jevosx.executor.keys import key_vocabulary
from jevosx.executor.safety import SafetyPolicy
from jevosx.logins import PASSWORD_SLOT, USERNAME_SLOT, LoginError, LoginStore, MemorySecrets, credential_slots
from jevosx.memory import HashingEmbedder, MemoryStore
from jevosx.router.policy import JevRouter, build_state
from jevosx.router.text import TextSource
from jevosx.sites import SiteError, host_matches, mask, normalize_site, page_host
from jevosx.types import TYPE_TEXT, Action, AppInfo
from tests.fakes import FakeDesktop, FakeNode, element, find_id, observation, scripted_client

SAFARI = AppInfo("Safari", "com.apple.Safari", pid=7)
SECRET = "correct horse battery staple"


@pytest.mark.parametrize(
    ("site", "host"),
    [
        ("github.com", "github.com"),
        ("https://www.GitHub.com/login?next=/", "github.com"),
        ("accounts.google.com", "accounts.google.com"),
        ("localhost:8080", "localhost"),
    ],
)
def test_normalize_site(site, host):
    assert normalize_site(site) == host


@pytest.mark.parametrize("site", ["", "github", "not a host", "https://", "exa mple.com"])
def test_normalize_site_rejects_non_hosts(site):
    with pytest.raises(SiteError):
        normalize_site(site)


@pytest.mark.parametrize(
    ("url", "host"),
    [
        ("https://github.com/login", "github.com"),
        ("http://github.com/login", None),  # never over plain http
        ("http://localhost:3000/login", "localhost"),
        ("file:///Users/me/login.html", None),
        ("javascript:alert(1)", None),
        (None, None),
    ],
)
def test_page_host(url, host):
    assert page_host(url) == host


@pytest.mark.parametrize(
    ("saved", "host", "ok"),
    [
        ("github.com", "github.com", True),
        ("github.com", "www.github.com", True),
        ("google.com", "accounts.google.com", True),
        ("github.com", "github.com.evil.io", False),
        ("github.com", "evilgithub.com", False),
        ("accounts.google.com", "google.com", False),
        ("me.github.io", "me.github.io", True),
        ("github.io", "attacker.github.io", False),  # shared hosting: exact host only
    ],
)
def test_host_matches(saved, host, ok):
    assert host_matches(saved, host) is ok


def test_mask():
    assert mask("matthew@gmail.com") == "m•••@gmail.com"
    assert mask("octocat") == "o•••t"
    assert mask("ab") == "•••"


def make_store(tmp_path):
    secrets = MemorySecrets()
    store = LoginStore(tmp_path / "logins.json", secrets)
    store.add("https://github.com/login", "octocat", SECRET)
    return store, secrets


def test_store_keeps_passwords_out_of_the_index(tmp_path):
    store, secrets = make_store(tmp_path)
    raw = (tmp_path / "logins.json").read_text()
    assert SECRET not in raw and json.loads(raw)["logins"][0]["host"] == "github.com"
    assert oct(os.stat(tmp_path / "logins.json").st_mode & 0o777) == "0o600"
    assert secrets.items == {("jevosx:github.com", "octocat"): SECRET}
    login = store.for_url("https://github.com/sessions/new")[0]
    assert store.password(login) == SECRET
    assert store.for_url("https://gitlab.com/") == [] and store.for_url("http://github.com/") == []
    store.add("github.com", "octocat", "new password")  # replaces, does not duplicate
    assert len(store.saved()) == 1 and secrets.items[("jevosx:github.com", "octocat")] == "new password"
    assert [x.username for x in store.remove("github.com")] == ["octocat"]
    assert store.saved() == [] and secrets.items == {}
    with pytest.raises(LoginError):
        store.add("not a site", "me", "pw")


def login_page(url="https://github.com/login", *, filled=False):
    return observation(
        [
            element(1, "AXTextField", "Username or email address", kind="text_input", ops=("TYPE_TEXT", "CLICK"),
                    in_web_area=True, value="octocat" if filled else None),
            element(2, "AXTextField", "Password", subrole="AXSecureTextField", kind="text_input",
                    ops=("TYPE_TEXT", "CLICK"), secure=True, in_web_area=True, filled=filled),
            element(3, "AXButton", "Sign in", in_web_area=True),
        ],
        app=SAFARI,
        window="Sign in to GitHub · GitHub",
        text="Sign in to GitHub" + (" · Signed in as octocat" if filled else ""),
    )  # fmt: skip


def with_url(obs, url):
    obs.page_url = url
    obs.fingerprint = obs.compute_fingerprint()
    return obs


def test_credential_slots_follow_the_page_and_read_the_keychain_only_when_typed(tmp_path):
    store, _ = make_store(tmp_path)
    reads = []
    original = store.password
    store.password = lambda login: reads.append(login) or original(login)  # type: ignore[method-assign]
    obs = with_url(login_page(), "https://github.com/login")
    slots = {s.name: s for s in credential_slots(store, obs, "log in to github")}
    assert set(slots) == {USERNAME_SLOT, PASSWORD_SLOT} and reads == []
    assert slots[USERNAME_SLOT].preview == "saved username for github.com: o•••t"
    assert slots[PASSWORD_SLOT].preview == "•••••• (saved password for github.com)" and slots[PASSWORD_SLOT].secret
    source = TextSource({})
    source.set_credentials(slots.values())
    resolved = source.resolve(PASSWORD_SLOT, goal="g", element=obs.elements[1], obs=obs, history=[])
    assert resolved.text == SECRET and resolved.secure_only and resolved.host == "github.com" and len(reads) == 1
    assert credential_slots(store, with_url(login_page(), "https://github.com.evil.io/login"), "log in") == []
    assert credential_slots(store, with_url(login_page(), "http://github.com/login"), "log in") == []
    no_form = with_url(observation([element(1, "AXLink", "Pricing")], app=SAFARI), "https://github.com/")
    assert credential_slots(store, no_form, "open the pricing page") == []  # nobody asked to sign in
    source.set_credentials([])
    assert source.slots == {}


@pytest.mark.parametrize(
    ("url", "field", "verdict"),
    [
        ("https://github.com/login", 2, "confirm"),
        ("https://github.com/login", 1, "deny"),  # a password never goes into a plain text field
        ("https://github.com.evil.io/login", 2, "deny"),
        ("http://github.com/login", 2, "deny"),
    ],
)
def test_safety_binds_saved_passwords_to_their_site(url, field, verdict):
    obs = login_page()
    action = Action(TYPE_TEXT, element=obs.elements[field - 1], text=SECRET, text_is_secret=True,
                    require_host="github.com", secure_only=True)  # fmt: skip
    assert SafetyPolicy().check(action, SAFARI, page_url=url).verdict == verdict


def test_safety_rejects_logins_outside_web_pages_and_can_skip_confirmation():
    obs = login_page()
    native = obs.elements[1]
    native.in_web_area = False
    action = Action(TYPE_TEXT, element=native, text=SECRET, text_is_secret=True, require_host="github.com",
                    secure_only=True)  # fmt: skip
    assert SafetyPolicy().check(action, SAFARI, page_url="https://github.com/login").verdict == "deny"
    native.in_web_area = True
    relaxed = SafetyPolicy(SafetySettings(confirm_credentials=False))
    assert relaxed.check(action, SAFARI, page_url="https://github.com/login").verdict == "allow"


def test_state_masks_saved_usernames_and_passwords(tmp_path):
    store, _ = make_store(tmp_path)
    obs = with_url(login_page(filled=True), "https://github.com/login?return_to=secret-token")
    source = TextSource({})
    source.set_credentials(credential_slots(store, obs, "log in to github.com"))
    history = [{"step": 1, "action": 'TYPE_TEXT [1] textfield "Username" ← (saved username for github.com)'}]
    state = build_state(obs, text_source=source, history=history)
    dumped = json.dumps(state, ensure_ascii=False)
    assert "octocat" not in dumped and SECRET not in dumped and "secret-token" not in dumped
    assert state["desktop"]["page"] == "https://github.com/login"
    assert state["elements"][1]["state"] == ["secure", "filled"]


class LoginDesktop(FakeDesktop):
    def __init__(self):
        screens = {
            "form": lambda: with_url(login_page(), "https://github.com/login"),
            "user": lambda: with_url(self._user_typed(), "https://github.com/login"),
            "both": lambda: with_url(login_page(filled=True), "https://github.com/login"),
            "2fa": lambda: with_url(
                observation(
                    [
                        element(
                            1,
                            "AXTextField",
                            "Authentication code",
                            kind="text_input",
                            ops=("TYPE_TEXT", "CLICK"),
                            in_web_area=True,
                        )
                    ],
                    app=SAFARI,
                    window="Two-factor authentication",
                    text="Enter the code from your app",
                ),
                "https://github.com/sessions/two-factor",
            ),  # fmt: skip
            "home": lambda: with_url(
                observation(
                    [element(1, "AXLink", "Repositories", in_web_area=True)],
                    app=SAFARI,
                    window="GitHub",
                    text="Signed in as octocat",
                ),
                "https://github.com/",
            ),  # fmt: skip
        }
        transitions = {
            ("form", "Username"): "user",
            ("user", "Password"): "both",
            ("both", "Sign in"): "2fa",
        }
        super().__init__(screens, "form", transitions)

    @staticmethod
    def _user_typed():
        obs = login_page()
        obs.elements[0].value = "octocat"
        return obs


def login_policy(body):
    labels = {e["label"]: e for e in body["state"]["elements"]}
    if "Repositories" in labels:
        return {"operation": "DONE"}
    if "Authentication code" in labels:
        return {"operation": "ASK_USER", "handoff_reason": "code"}
    if not labels["Username or email address"].get("value"):
        return {
            "operation": "TYPE_TEXT",
            "type_text_target": find_id(body, "type_text_target", "Username"),
            "text_slot": USERNAME_SLOT,
        }
    if "filled" not in labels["Password"].get("state", []):
        return {
            "operation": "TYPE_TEXT",
            "type_text_target": find_id(body, "type_text_target", "Password"),
            "text_slot": PASSWORD_SLOT,
        }
    return {"operation": "CLICK", "click_target": find_id(body, "click_target", "Sign in")}


def run_login(tmp_path, *, approve=True, handoff=lambda request, obs: True, settings=None):
    store, _ = make_store(tmp_path)
    desktop = LoginDesktop()
    requests: list[dict] = []
    confirmations: list[str] = []
    handoffs: list[str] = []
    settings = settings or Settings()
    settings.agent.fallback_log = ""
    settings.memory.store_typed_text = True  # even then, saved logins must not be stored
    memory = MemoryStore(tmp_path / "memory.db", HashingEmbedder(settings.memory.dim))

    def confirm(action, reason):
        confirmations.append(reason)
        return approve

    def hand_off(request, obs):
        handoffs.append(request)
        done = handoff(request, obs)
        if done:
            desktop.screen = "home"  # the person typed the code from their phone
        return done

    agent = Agent(
        observer=desktop,
        executor=desktop,
        router=JevRouter(scripted_client(login_policy, requests), keys=key_vocabulary()),
        settings=settings,
        memory=memory,
        logins=store,
        confirm=confirm,
        handoff=hand_off,
        sleep=lambda _s: None,
    )
    result = agent.run("Log in to github.com")
    return result, desktop, requests, confirmations, handoffs, memory


def test_agent_signs_in_with_the_saved_login_and_hands_off_the_code(tmp_path):
    result, desktop, requests, confirmations, handoffs, memory = run_login(tmp_path)
    assert result.status == "done", result.message
    assert desktop.executed == [
        'TYPE_TEXT [1] textfield "Username or email address" <- octocat',
        f'TYPE_TEXT [2] password field "Password" <- {SECRET}',
        'CLICK [3] button "Sign in"',
    ]
    assert confirmations == ["type your saved password for github.com into this field"]
    assert handoffs == ["type a verification, two-factor or one-time code (Safari, window “Two-factor authentication”)"]
    assert [e.status for e in result.events] == ["acted", "acted", "acted", "handoff", "done"]
    everything = json.dumps(requests, ensure_ascii=False) + json.dumps(
        [e.to_dict() for e in result.events], default=str
    )
    assert SECRET not in everything and "octocat" not in everything
    export = tmp_path / "export.jsonl"
    memory.export_jsonl(export)
    assert SECRET not in export.read_text() and "octocat" not in export.read_text()
    assert "ASK_USER" in requests[-2]["questions"]["operation"]["criteria"]


def test_declined_password_is_not_typed(tmp_path):
    result, desktop, *_ = run_login(tmp_path, approve=False)
    assert not any(SECRET in line for line in desktop.executed)
    assert any(e.status == "declined" for e in result.events)


def test_stopping_during_a_handoff_ends_the_run(tmp_path):
    result, *_ = run_login(tmp_path, handoff=lambda request, obs: False)
    assert result.status == "aborted" and "hand-off" in result.message


def test_ask_user_is_offered_only_with_a_handoff_callback():
    obs = login_page()
    router = JevRouter(scripted_client(lambda body: {}), keys=key_vocabulary())
    assert "ASK_USER" not in router.space(obs, TextSource({}), "log in").operations
    space = router.space(obs, TextSource({}), "log in", handoff=True)
    assert "ASK_USER" in space.operations and set(space.targets_for("ASK_USER")) >= {"code", "captcha", "approve"}


def test_executor_rechecks_the_fields_own_page_before_typing():
    from jevosx.executor.mac import MacExecutor

    executor = object.__new__(MacExecutor)  # the check needs no Accessibility connection
    web = FakeNode("AXWebArea", URL="https://github.com/login")
    field = element(2, "AXTextField", "Password", secure=True, in_web_area=True,
                    node=FakeNode("AXTextField", Parent=FakeNode("AXGroup", Parent=web)))  # fmt: skip
    action = Action(TYPE_TEXT, element=field, text=SECRET, text_is_secret=True, require_host="github.com",
                    secure_only=True)  # fmt: skip
    assert executor._check_credential(action) is None
    web.attrs["AXURL"] = "https://github-login.example.com/"
    refused = executor._check_credential(action)
    assert refused is not None and not refused.ok and "nothing typed" in refused.detail
    field.secure = False
    assert "password field" in executor._check_credential(action).detail
