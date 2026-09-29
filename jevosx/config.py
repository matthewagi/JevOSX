"""Typed settings from TOML, a local .env file and environment variables (later sources override earlier ones)."""

from __future__ import annotations

import dataclasses
import os
import tomllib
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

from .errors import ConfigError

DEFAULT_CONFIG_PATHS = (Path("jevosx.toml"), Path("~/.config/jevosx/config.toml"))
# The checkout this package runs from (e.g. ~/JevOSX for the installer's editable install), if any.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def dotenv_candidates() -> list[Path]:
    """Where .env files are read from, first match wins per variable: the current folder, the JevOSX checkout
    (so the global `jevosx` command finds the installer's key from any folder), then ~/.config/jevosx/.env."""
    candidates = [Path.cwd() / ".env"]
    if (PROJECT_ROOT / "pyproject.toml").is_file():
        candidates.append(PROJECT_ROOT / ".env")
    candidates.append(Path("~/.config/jevosx/.env").expanduser())
    unique: list[Path] = []
    for candidate in candidates:
        if candidate.resolve() not in {u.resolve() for u in unique}:
            unique.append(candidate)
    return unique


@dataclass
class JevSettings:
    api_key_env: str = "TYPESAFE_API_KEY"
    endpoint: str = "https://api.typesafe.ai/v1/systemone"
    model: str = "jev-latest"
    timeout_s: float = 10.0
    max_retries: int = 2
    http2: bool = True
    max_choices: int = 200  # Jev accepts up to 255 options per choice question
    # Context guardrails: what the state payload may contain.
    max_state_bytes: int = 48_000  # hard cap on the serialized state; text is trimmed first, then elements
    include_disabled: bool = False  # disabled / non-interactive elements are not sent unless enabled here
    offer_installed_apps: str = "mentioned"  # mentioned | all | none: non-running apps offered to OPEN_APP

    def api_key(self, env: Mapping[str, str] | None = None) -> str | None:
        return (env if env is not None else os.environ).get(self.api_key_env) or None


@dataclass
class ObserverSettings:
    max_nodes: int = 3000
    max_elements: int = 180
    max_depth: int = 64
    max_children: int = 400
    time_budget_s: float = 1.5
    max_text_chars: int = 2000
    probe_generic: bool = True
    include_menus: bool = True
    max_menu_items: int = 200
    menu_cache_ttl_s: float = 3.0
    include_installed_apps: bool = True
    app_dirs: list[str] = field(
        default_factory=lambda: [
            "/Applications",
            "/Applications/Utilities",
            "/System/Applications",
            "/System/Applications/Utilities",
            "~/Applications",
        ]
    )
    messaging_timeout_s: float = 0.5
    enable_web_accessibility: bool = True
    # Apps that draw their own interface: OCR the focused window when Accessibility sees almost nothing there.
    vision: str = "auto"  # auto | always | off (needs the Screen Recording permission)
    vision_min_controls: int = 4  # auto: fewer interactive elements than this (and hardly any text) → OCR
    vision_max_items: int = 60
    vision_min_confidence: float = 0.35


@dataclass
class ExecutorSettings:
    typing_mode: str = "auto"  # auto | ax | keys
    pointer_fallback: bool = False  # last-resort synthetic click at the element's AX frame centre
    settle_timeout_s: float = 0.8
    settle_poll_s: float = 0.05
    wait_s: float = 0.6
    launch_timeout_s: float = 8.0
    key_delay_s: float = 0.006
    scroll_page_fraction: float = 0.8
    focus_timeout_s: float = 1.5  # how long to wait for a window to come forward before keys are sent to it


@dataclass
class MemorySettings:
    enabled: bool = True
    path: str = "~/.jevosx/memory.db"
    dim: int = 512
    max_hints: int = 4
    top_episodes: int = 8
    min_goal_similarity: float = 0.3
    min_state_similarity: float = 0.35
    store_typed_text: bool = False
    keep_episodes: int = 5000


@dataclass
class SafetySettings:
    confirm_patterns: list[str] = field(
        default_factory=lambda: [
            r"\bdelete\b",
            r"\bsend\b",
            r"\berase\b",
            r"\btrash\b",
            r"\bdiscard\b",
            r"\bempty\b",
            r"\buninstall\b",
            r"\bshut ?down\b",
            r"\brestart\b",
            r"\blog ?out\b",
            r"\bsign ?out\b",
            r"\bpurchase\b",
            r"\bbuy\b",
            r"\bpay\b",
            r"\bplace order\b",
            r"\btransfer\b",
            r"\bfactory reset\b",
            r"\bformat\b",
            r"\bpublish\b",
            r"\bpost\b",
            r"\bsubmit\b",
        ]
    )
    deny_apps: list[str] = field(default_factory=lambda: ["com.apple.keychainaccess", "com.apple.Passwords"])
    confirm_keys: list[str] = field(default_factory=lambda: ["CMD_Q"])
    deny_keys: list[str] = field(default_factory=list)
    confirm_all: bool = False
    confirm_credentials: bool = True  # ask before typing a saved password (see [logins])


@dataclass
class AgentSettings:
    max_steps: int = 40
    # Confidence gate: an action (or DONE) is only executed when Jev's confidence for the operation AND for the
    # chosen target are at least this floor. Otherwise the fallback policy runs and nothing is executed blindly.
    min_confidence: float = 0.65
    # Steps that are easily undone need less (see jevosx/risk.py): opening windows, switching apps, scrolling,
    # typing into a field (safe), and clicks, Return, menu commands, DONE (routine). Never above min_confidence.
    safe_confidence: float = 0.35
    routine_confidence: float = 0.5
    low_confidence_policy: str = "retry"  # retry (re-observe) | ask (human approves) | stop
    # In the web console someone is watching, so an unsure step is shown for approval instead of re-asked.
    console_low_confidence_policy: str = "ask"
    max_low_confidence_retries: int = 2
    # With "ask": look again this many times before asking you. An unsure moment is often a page still loading.
    ask_after_retries: int = 1
    # From the web console's own window the agent may only open a window/tab or switch apps/windows. Those moves
    # change nothing, so by default they are not held back by the floor (set true to gate them too).
    gate_console_navigation: bool = False
    fallback_log: str = "~/.jevosx/fallbacks.jsonl"  # JSON-lines record of every withheld decision ("" = off)
    max_stale_retries: int = 3
    stuck_after: int = 3
    history_size: int = 8
    max_done_rejections: int = 2
    max_handoffs: int = 3  # ASK_USER hand-offs (2FA codes, CAPTCHAs…) per run
    plan: str = "auto"  # auto: the writer's model reads the goal once per run (steps + values to type) | off
    # Work behind your window: the agent keeps its own work window and reads it where it is. Clicks and field writes
    # go through Accessibility without bringing it forward; for key presses it comes forward briefly, and then
    # whatever you were using (the console, Terminal…) is brought back. false: the agent works in front, as before.
    background: bool = True


@dataclass
class TextModelSettings:
    """Optional OpenAI-compatible model that writes field text when no text slot fits. Off until `model` is set."""

    api_key_env: str = "TEXT_MODEL_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = ""
    timeout_s: float = 15.0

    def api_key(self, env: Mapping[str, str] | None = None) -> str | None:
        return (env if env is not None else os.environ).get(self.api_key_env) or None


@dataclass
class WriterSettings:
    """Who composes free-form text for goals like "write a poem" (Jev never writes text itself)."""

    backend: str = "auto"  # auto ([text_model] if configured, else Apple's on-device model) | apple | openai | off
    offer: str = "auto"  # auto: offer GENERATE only when the goal asks for new text | always
    temperature: float = 0.7
    max_tokens: int = 800
    timeout_s: float = 60.0
    helper_dir: str = "~/.jevosx/bin"  # where the compiled Apple helper lives


@dataclass
class LoginSettings:
    """Saved website logins (`jevosx login add github.com`): passwords in the macOS Keychain, never in files."""

    enabled: bool = True
    index_path: str = "~/.jevosx/logins.json"  # hosts and usernames only


@dataclass
class KeySettings:
    custom: dict[str, str] = field(default_factory=dict)  # e.g. {"SEND" = "cmd+shift+d"}
    disabled: list[str] = field(default_factory=list)


@dataclass
class Settings:
    jev: JevSettings = field(default_factory=JevSettings)
    observer: ObserverSettings = field(default_factory=ObserverSettings)
    executor: ExecutorSettings = field(default_factory=ExecutorSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    safety: SafetySettings = field(default_factory=SafetySettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    text_model: TextModelSettings = field(default_factory=TextModelSettings)
    writer: WriterSettings = field(default_factory=WriterSettings)
    logins: LoginSettings = field(default_factory=LoginSettings)
    keys: KeySettings = field(default_factory=KeySettings)

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        *,
        env: MutableMapping[str, str] | None = None,
        dotenv: str | Path | None = "auto",
    ) -> Settings:
        """`dotenv="auto"` reads every file from dotenv_candidates(); a path reads just that file; None reads none."""
        env = os.environ if env is None else env
        if dotenv == "auto":
            for candidate in dotenv_candidates():
                load_dotenv(candidate, env)
        elif dotenv:
            load_dotenv(dotenv, env)
        data: dict[str, Any] = {}
        config_path = resolve_config_path(path, env)
        if config_path is not None:
            try:
                data = tomllib.loads(config_path.read_text())
            except (OSError, tomllib.TOMLDecodeError) as exc:
                raise ConfigError(f"Cannot read config {config_path}: {exc}") from exc
        settings = _build(cls, data, "config")
        settings.apply_env(env)
        settings.validate()
        return settings

    def apply_env(self, env: Mapping[str, str]) -> None:
        overrides: dict[str, tuple[object, str, type]] = {
            "JEV_MODEL": (self.jev, "model", str),
            "TYPESAFE_ENDPOINT": (self.jev, "endpoint", str),
            "JEVOSX_MEMORY_PATH": (self.memory, "path", str),
            "JEVOSX_MAX_STEPS": (self.agent, "max_steps", int),
            "TEXT_MODEL": (self.text_model, "model", str),
            "TEXT_MODEL_BASE_URL": (self.text_model, "base_url", str),
            "JEVOSX_WRITER": (self.writer, "backend", str),
        }
        for name, (section, attr, kind) in overrides.items():
            if env.get(name):
                try:
                    setattr(section, attr, kind(env[name]))
                except ValueError as exc:
                    raise ConfigError(f"{name}={env[name]!r} is not a valid {kind.__name__}") from exc

    @property
    def memory_path(self) -> Path:
        return Path(self.memory.path).expanduser()

    def validate(self) -> None:
        choices = {
            "agent.low_confidence_policy": (self.agent.low_confidence_policy, ("retry", "ask", "stop")),
            "agent.console_low_confidence_policy": (self.agent.console_low_confidence_policy, ("retry", "ask", "stop")),
            "jev.offer_installed_apps": (self.jev.offer_installed_apps, ("mentioned", "all", "none")),
            "executor.typing_mode": (self.executor.typing_mode, ("auto", "ax", "keys")),
            "writer.backend": (self.writer.backend, ("auto", "apple", "openai", "off")),
            "writer.offer": (self.writer.offer, ("auto", "always")),
            "observer.vision": (self.observer.vision, ("auto", "always", "off")),
            "agent.plan": (self.agent.plan, ("auto", "off")),
        }
        for name, (value, allowed) in choices.items():
            if value not in allowed:
                raise ConfigError(f"{name} must be one of {', '.join(allowed)} (got {value!r})")
        for name in ("min_confidence", "safe_confidence", "routine_confidence"):
            if not 0.0 <= getattr(self.agent, name) <= 1.0:
                raise ConfigError(f"agent.{name} must be between 0 and 1")
        if self.agent.ask_after_retries < 0:
            raise ConfigError("agent.ask_after_retries must be 0 or more")
        if not 2 <= self.jev.max_choices <= 255:
            raise ConfigError("jev.max_choices must be between 2 and 255")


def resolve_config_path(path: str | Path | None, env: Mapping[str, str]) -> Path | None:
    if path is not None:
        candidate = Path(path).expanduser()
        if not candidate.is_file():
            raise ConfigError(f"Config file not found: {candidate}")
        return candidate
    if env.get("JEVOSX_CONFIG"):
        return resolve_config_path(env["JEVOSX_CONFIG"], {})
    return next((p.expanduser() for p in DEFAULT_CONFIG_PATHS if p.expanduser().is_file()), None)


def load_dotenv(path: str | Path, env: MutableMapping[str, str]) -> None:
    """Minimal KEY=VALUE loader. Existing environment variables always win."""
    file = Path(path).expanduser()
    if not file.is_file():
        return
    for raw in file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        env.setdefault(key, value)


def _build(kind: type, data: Mapping[str, Any], where: str) -> Any:
    """Construct a settings dataclass, rejecting unknown keys so typos fail loudly."""
    if not isinstance(data, Mapping):
        raise ConfigError(f"[{where}] must be a table")
    hints = get_type_hints(kind)
    names = {f.name for f in dataclasses.fields(kind)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"Unknown key(s) in [{where}]: {', '.join(sorted(unknown))}")
    values: dict[str, Any] = {}
    for name, raw in data.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            values[name] = _build(hint, raw, name)  # type: ignore[arg-type]
            continue
        values[name] = _coerce(raw, hint, f"{where}.{name}")
    return kind(**values)


def _coerce(value: Any, hint: Any, where: str) -> Any:
    origin = getattr(hint, "__origin__", hint)
    if origin is float and isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if origin in (int, str, bool) and type(value) is origin:
        return value
    if origin is list and isinstance(value, list):
        return list(value)
    if origin is dict and isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    raise ConfigError(f"{where} has the wrong type ({type(value).__name__})")
