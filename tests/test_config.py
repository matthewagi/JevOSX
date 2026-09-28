import pytest

from jevosx.config import Settings
from jevosx.errors import ConfigError


def test_defaults_toml_env_and_dotenv(tmp_path):
    config = tmp_path / "jevosx.toml"
    config.write_text(
        """
[jev]
model = "jev-1.13.0"
timeout_s = 3

[memory]
path = "/tmp/mem.db"

[keys.custom]
SEND = "cmd+shift+d"
"""
    )
    dotenv = tmp_path / ".env"
    dotenv.write_text('export TYPESAFE_API_KEY="sk-from-dotenv"\nJEVOSX_MAX_STEPS=7\n# comment\n')
    env = {"TYPESAFE_API_KEY": "sk-from-env"}
    settings = Settings.load(config, env=env, dotenv=dotenv)
    assert settings.jev.model == "jev-1.13.0" and settings.jev.timeout_s == 3.0
    assert settings.jev.api_key(env) == "sk-from-env"  # real environment wins over .env
    assert settings.agent.max_steps == 7
    assert settings.keys.custom == {"SEND": "cmd+shift+d"}
    assert settings.observer.max_elements == 180  # untouched default


def test_unknown_keys_and_wrong_types_fail_loudly(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[jev]\nmodle = 'typo'\n")
    with pytest.raises(ConfigError, match="modle"):
        Settings.load(bad, env={}, dotenv=None)
    bad.write_text("[agent]\nmax_steps = 'ten'\n")
    with pytest.raises(ConfigError, match="max_steps"):
        Settings.load(bad, env={}, dotenv=None)
    with pytest.raises(ConfigError, match="not found"):
        Settings.load(tmp_path / "missing.toml", env={}, dotenv=None)


def test_example_config_is_valid():
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "config" / "jevosx.example.toml"
    settings = Settings.load(example, env={}, dotenv=None)
    assert settings.jev.endpoint.startswith("https://")
