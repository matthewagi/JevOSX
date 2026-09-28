"""JevOSX: a coordinate-free, OCR-free macOS automation agent.

Accessibility trees in → TypeSafe Jev typed decisions → deterministic AX actions → local trajectory memory.
"""

from .agent import Agent, RunResult, StepEvent
from .config import Settings
from .types import Action, ActionResult, AppInfo, Observation, UIElement

__version__ = "0.1.0"

__all__ = [
    "Action",
    "ActionResult",
    "Agent",
    "AppInfo",
    "Observation",
    "RunResult",
    "Settings",
    "StepEvent",
    "UIElement",
    "__version__",
]
