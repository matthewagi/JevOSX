"""Saved website logins: passwords in the macOS Keychain, bound to one site, typed only into password fields.

`jevosx login add github.com` stores the password in the login Keychain (service "jevosx:github.com") and lists
the host and username in ~/.jevosx/logins.json. The index never holds passwords.

During a run, a saved login becomes two text slots only while the page on screen is that site over https:
`login_username` and `login_password`. Jev sees masked previews; the password never reaches Jev, the writer,
memory, logs or run history, and the executor re-checks the page URL right before typing it.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .errors import JevOSXError
from .router.text import TextSlot
from .sites import SiteError, host_matches, mask, normalize_site, page_host
from .types import Observation

SERVICE_PREFIX = "jevosx:"
USERNAME_SLOT = "login_username"
PASSWORD_SLOT = "login_password"
_LOGIN_GOAL = re.compile(r"\b(?:log\s*-?\s*in|sign\s*-?\s*in|login|signin|authenticate|log\s+me\s+in)\b", re.I)


class LoginError(JevOSXError):
    """A login could not be saved, found or removed."""


def wants_login(goal: str) -> bool:
    return bool(_LOGIN_GOAL.search(goal))


@dataclass(frozen=True, slots=True)
class Login:
    host: str
    username: str
    added: float = 0.0

    @property
    def service(self) -> str:
        return SERVICE_PREFIX + self.host


class SecretBackend(Protocol):
    def set_password(self, service: str, username: str, password: str) -> None: ...

    def get_password(self, service: str, username: str) -> str | None: ...

    def delete_password(self, service: str, username: str) -> None: ...


class MemorySecrets:
    """In-memory backend for tests and the demo console."""

    def __init__(self) -> None:
        self.items: dict[tuple[str, str], str] = {}

    def set_password(self, service: str, username: str, password: str) -> None:
        self.items[(service, username)] = password

    def get_password(self, service: str, username: str) -> str | None:
        return self.items.get((service, username))

    def delete_password(self, service: str, username: str) -> None:
        self.items.pop((service, username), None)


def keychain() -> SecretBackend:
    """The macOS login Keychain (via keyring's Security-framework backend)."""
    try:
        from keyring.backends.macOS import Keyring
    except ImportError as exc:
        raise LoginError("saving logins needs the 'keyring' package: pip install keyring") from exc
    backend = Keyring()
    try:
        _ = backend.priority  # raises off macOS or without the Security API
    except RuntimeError as exc:
        raise LoginError(f"the macOS Keychain is unavailable: {exc}") from exc
    return backend


class LoginStore:
    def __init__(self, index_path: str | Path, secrets: SecretBackend | None = None):
        self.index_path = Path(index_path).expanduser()
        self._secrets = secrets

    @property
    def secrets(self) -> SecretBackend:
        if self._secrets is None:
            self._secrets = keychain()
        return self._secrets

    # ---- index ------------------------------------------------------------------------------------------------
    def saved(self) -> list[Login]:
        try:
            data = json.loads(self.index_path.read_text())
        except FileNotFoundError:
            return []
        except (OSError, ValueError) as exc:
            raise LoginError(f"cannot read {self.index_path}: {exc}") from exc
        logins = []
        for item in data.get("logins", []) if isinstance(data, dict) else []:
            with contextlib.suppress(KeyError, TypeError, ValueError):
                logins.append(Login(str(item["host"]), str(item["username"]), float(item.get("added", 0))))
        return logins

    def _write(self, logins: Iterable[Login]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"logins": [{"host": x.host, "username": x.username, "added": x.added} for x in logins]}
        fd, tmp = tempfile.mkstemp(prefix=".logins-", dir=self.index_path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(payload, out, indent=2)
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.index_path)

    # ---- commands ---------------------------------------------------------------------------------------------
    def add(self, site: str, username: str, password: str) -> Login:
        host = _host(site)
        username = username.strip()
        if not username or not password:
            raise LoginError("both a username and a password are needed")
        login = Login(host, username, time.time())
        self.secrets.set_password(login.service, username, password)
        others = [x for x in self.saved() if (x.host, x.username) != (host, username)]
        self._write([*others, login])
        return login

    def remove(self, site: str, username: str | None = None) -> list[Login]:
        host = _host(site)
        logins = self.saved()
        gone = [x for x in logins if x.host == host and (username is None or x.username == username)]
        for login in gone:
            with contextlib.suppress(Exception):  # already missing from the Keychain is fine
                self.secrets.delete_password(login.service, login.username)
        self._write([x for x in logins if x not in gone])
        return gone

    def for_url(self, url: str | None) -> list[Login]:
        """Saved logins usable on this page, most specific host first, newest first."""
        host = page_host(url)
        if host is None:
            return []
        matches = [x for x in self.saved() if host_matches(x.host, host)]
        return sorted(matches, key=lambda x: (-len(x.host), -x.added))

    def password(self, login: Login) -> str | None:
        return self.secrets.get_password(login.service, login.username)


def _host(site: str) -> str:
    try:
        return normalize_site(site)
    except SiteError as exc:
        raise LoginError(str(exc)) from None


def credential_slots(store: LoginStore, obs: Observation, goal: str) -> list[TextSlot]:
    """Username + password slots for the page on screen, when a login is saved for it and the goal or the page
    calls for signing in (a visible password field). The password is read from the Keychain only when typed."""
    if not (wants_login(goal) or any(e.secure for e in obs.elements)):
        return []
    try:
        logins = store.for_url(obs.page_url)
    except LoginError:
        return []
    if not logins:
        return []
    lowered = goal.lower()
    login = next((x for x in logins if x.username.lower() in lowered), logins[0])
    return [
        TextSlot(
            USERNAME_SLOT,
            login.username,
            host=login.host,
            label=f"saved username for {login.host}: {mask(login.username)}",
        ),
        TextSlot(
            PASSWORD_SLOT,
            "",
            host=login.host,
            label=f"saved password for {login.host}",
            secure_only=True,
            fetch=lambda: store.password(login),
        ),
    ]
