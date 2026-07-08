"""Optional sandbox backends for PraisonAI (praisonai.sandbox entry points)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .capsule import CapsuleSandbox

__all__ = ["CapsuleSandbox"]


def __getattr__(name: str) -> Any:
    if name == "CapsuleSandbox":
        from .capsule import CapsuleSandbox

        return CapsuleSandbox
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
