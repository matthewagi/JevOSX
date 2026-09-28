"""Free-form text writers: who composes TYPE_TEXT content when a goal asks for new text ("write a poem").

Backends, chosen by `[writer] backend`:
- `apple`: Apple's on-device model through a small compiled Swift helper (macOS 26+, Apple Intelligence).
- `openai`: an OpenAI-compatible chat model configured in `[text_model]`.
- `auto` (default): the `[text_model]` model if one is configured, otherwise Apple's on-device model.
Jev never writes text; it chooses the field and whether to type, and the writer only fills in the words.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

from .base import TextWriter, WriterStatus, clean_generated, wants_generation, writer_prompt

if TYPE_CHECKING:
    from ..config import Settings

__all__ = ["TextWriter", "WriterStatus", "clean_generated", "create_writer", "wants_generation", "writer_prompt"]


def create_writer(
    settings: Settings,
    *,
    notify: Callable[[str], None] | None = None,
    platform: str = sys.platform,
    rebuild: bool = False,
    backend: str | None = None,
) -> tuple[TextWriter | None, WriterStatus]:
    """Build the configured writer. Returns (None, status-with-reason) when none is usable; never raises."""
    choice = backend or settings.writer.backend
    if choice == "off":
        return None, WriterStatus("none", False, "disabled", "writer.backend = off")
    text_model = settings.text_model
    key = text_model.api_key()
    openai_ready = bool(text_model.model and key)
    if choice == "openai" or (choice == "auto" and openai_ready):
        if not openai_ready:
            missing = "model" if not text_model.model else text_model.api_key_env
            return None, WriterStatus(
                "openai", False, "notConfigured", f"[text_model] {missing} is not set", "see config/jevosx.example.toml"
            )
        from .openai import LLMTextWriter

        writer = LLMTextWriter(
            base_url=text_model.base_url, api_key=key or "", model=text_model.model, timeout_s=text_model.timeout_s
        )
        return writer, WriterStatus("openai", True, "available", text_model.model)
    from .apple import AppleWriter

    cfg = settings.writer
    return AppleWriter.prepare(
        cfg.helper_dir,
        rebuild=rebuild,
        platform=platform,
        notify=notify,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout_s=cfg.timeout_s,
    )
