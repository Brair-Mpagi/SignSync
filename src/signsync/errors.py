"""Exception types shared across the package."""

from __future__ import annotations


class SignSyncError(Exception):
    """Base class for every error this package raises deliberately."""


class MissingDependencyError(SignSyncError):
    """An optional extra is required for the requested backend.

    Raised instead of a bare ``ImportError`` so the message can name the extra to
    install and say what still works without it.
    """

    def __init__(self, feature: str, extra: str, package: str) -> None:
        self.feature = feature
        self.extra = extra
        self.package = package
        super().__init__(
            f"{feature} needs the optional package {package!r}. "
            f'Install it with: pip install -e ".[{extra}]"'
        )


class ConsentError(SignSyncError):
    """A clip was requested whose participant consent does not permit the use.

    This is deliberately an error and not a warning: silently dropping the clip
    would hide a compliance problem, and silently using it would create one
    (plan §16.1).
    """


class CorpusError(SignSyncError):
    """The corpus on disk is inconsistent with the declared schema."""


class SplitError(SignSyncError):
    """A requested train/test split would violate signer independence (plan §14)."""
