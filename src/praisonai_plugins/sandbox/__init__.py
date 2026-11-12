"""Optional sandbox backends for PraisonAI (praisonai.sandbox entry points)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .boxd import BoxdSandbox
    from .capsule import CapsuleSandbox

__all__ = ["BoxdSandbox", "CapsuleSandbox"]


def __getattr__(name: str) -> Any:
    if name == "CapsuleSandbox":
        from .capsule import CapsuleSandbox

        return CapsuleSandbox
    if name == "BoxdSandbox":
        from .boxd import BoxdSandbox

        return BoxdSandbox
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
