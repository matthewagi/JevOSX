"""Exception hierarchy. Every failure mode the agent loop reacts to has its own type."""


class JevOSXError(Exception):
    """Base class for all framework errors."""


class ConfigError(JevOSXError):
    """Invalid or unreadable configuration."""


class PlatformError(JevOSXError):
    """The macOS backends were requested on a platform (or Python env) that cannot provide them."""


class AccessibilityPermissionError(JevOSXError):
    """The host process is not trusted for Accessibility (System Settings › Privacy & Security)."""


class AXError(JevOSXError):
    """An Accessibility API call returned a non-success AXError code."""

    def __init__(self, code: int, operation: str):
        self.code = code
        self.operation = operation
        super().__init__(f"{operation} failed with AXError {code}")


class StaleElementError(JevOSXError):
    """The observed element or app changed between observation and execution. Re-observe, never guess."""


class JevError(JevOSXError):
    """The Jev endpoint failed or returned something that cannot drive an action."""

    def __init__(self, message: str, *, status: int | None = None):
        self.status = status
        super().__init__(message)


class JevAuthError(JevError):
    """Missing, invalid or unauthorized API key."""


class JevResponseError(JevError):
    """A structurally invalid answer. No action is executed from it."""


class RouterContractError(JevOSXError):
    """A request would break the strict-choice contract (open-ended question, or a target id missing from the
    element table). Raised before anything is sent."""


class LowConfidenceError(JevOSXError):
    """Jev's confidence is below the configured floor. The decision is withheld; the fallback policy decides next."""

    def __init__(self, decision: object, confidence: float, floor: float):
        self.decision = decision
        self.confidence = confidence
        self.floor = floor
        super().__init__(f"confidence {confidence:.2f} is below the floor {floor:.2f}; action withheld")


class TextUnavailableError(JevOSXError):
    """TYPE_TEXT was chosen but no text source can supply a value."""
