"""
Feedo integration plugin for PraisonAI Agents.

Registers Feedo — a decentralized storage + semantic search network — as a
memory backend for PraisonAI agents. Your identity is your crypto wallet
(``did:feedo:0x...``) — no accounts, no KYC, no vendor lock-in.

This plugin is self-contained: it ships the ``FeedoMemoryAdapter`` (the
``MemoryProtocol`` implementation) and registers it with the core memory
registry on ``ON_INIT``. The ``feedo-sdk`` dependency is imported lazily, only
when an agent actually selects ``provider: "feedo"``, so installing this plugin
never forces the SDK on users who don't use Feedo.

Configuration (in the agent ``memory`` dict)::

    from praisonaiagents import Agent

    agent = Agent(
        name="Assistant",
        memory={
            "provider": "feedo",
            "config": {
                "usage_key": "0x...",    # delegated usage key (only this is required)
                "user_id": "user123",    # optional: isolate memories per user
                "private": True,         # optional: private (default) or public
            },
        },
    )

The owner ``did`` is auto-resolved from the usage key's delegation, so it does
not need to be provided explicitly.
"""

from typing import Any

from praisonaiagents._logging import get_logger
from praisonaiagents.plugins.plugin import Plugin, PluginHook, PluginInfo

logger = get_logger(__name__)


class FeedoMemoryAdapter:
    """Memory adapter backed by the Feedo decentralized search network.

    Implements the PraisonAI ``MemoryProtocol`` (short/long-term store + search)
    plus the optional delete/reset helpers.
    """

    def __init__(self, **kwargs: Any) -> None:
        # PraisonAI's Memory class passes the user config inside a nested
        # "config" dict (e.g. {"provider": "feedo", "config": {"usage_key": ...}}),
        # but a direct caller may also pass the keys at the top level. Support both.
        nested = kwargs.get("config") or {}

        def _cfg(key: str, default: Any = None) -> Any:
            if key in kwargs:
                return kwargs[key]
            return nested.get(key, default)

        # Fail fast on a missing identity before attempting to import the SDK,
        # so a config mistake surfaces with remediation guidance rather than a
        # confusing "module not found" error.
        if not (_cfg("usage_key") or _cfg("private_key")):
            raise ValueError(
                "Feedo memory backend requires a `usage_key` (delegated usage key) "
                "or `private_key`. Register a DID and generate a usage key with "
                "`feedo init`, then pass it as memory.config.usage_key."
            )

        try:
            from feedo import FeedoMemory
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "feedo-sdk is not installed. Run: pip install feedo-sdk>=0.1.24"
            ) from e

        self._memory = FeedoMemory(
            usage_key=_cfg("usage_key"),
            private_key=_cfg("private_key"),
            did=_cfg("did"),
            user_id=_cfg("user_id"),
            private=_cfg("private", True),
            search_seeds=_cfg("search_seeds"),
            consensus_seeds=_cfg("consensus_seeds"),
            storage_seeds=_cfg("storage_seeds"),
        )

    # ------------------------------------------------------------------
    # MemoryProtocol
    # ------------------------------------------------------------------

    def store_short_term(
        self, text: str, metadata: dict[str, Any] | None = None, **kwargs: Any
    ) -> str:
        """Store content in short-term memory. Returns the memory id."""
        return self._memory.add_short(text, metadata)

    def search_short_term(
        self, query: str, limit: int = 5, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Search short-term memory."""
        return self._memory.search_short(query, limit=limit)

    def store_long_term(
        self, text: str, metadata: dict[str, Any] | None = None, **kwargs: Any
    ) -> str:
        """Store content in long-term memory. Returns the memory id."""
        return self._memory.add_long(text, metadata)

    def search_long_term(
        self, query: str, limit: int = 5, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Search long-term memory."""
        return self._memory.search_long(query, limit=limit)

    def get_all_memories(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Return all stored memories (short + long)."""
        return self._memory.get_all_memories()

    # ------------------------------------------------------------------
    # DeletableMemoryProtocol
    # ------------------------------------------------------------------

    def delete_memory(
        self, memory_id: str, tier: str | None = None, **kwargs: Any
    ) -> bool:
        """Delete a specific memory by id.

        Feedo currently supports namespace-level deletion only; per-item delete
        is not available yet, so this returns ``False``.
        """
        return False

    # ------------------------------------------------------------------
    # ResettableMemoryProtocol
    # ------------------------------------------------------------------

    def reset_short_term(self) -> None:
        """Clear all short-term memories."""
        self._memory.clear_short()

    def reset_long_term(self) -> None:
        """Clear all long-term memories."""
        self._memory.clear_long()


def create_feedo_memory_adapter(**kwargs: Any) -> FeedoMemoryAdapter:
    """Factory function for the Feedo memory adapter (lazy-loads feedo-sdk)."""
    return FeedoMemoryAdapter(**kwargs)


class FeedoPlugin(Plugin):
    """Registers Feedo as a memory backend for PraisonAI agents."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="feedo",
            version="1.0.0",
            description=(
                "Feedo — decentralized storage + semantic search memory backend "
                "(wallet identity, E2E-encrypted, free testnet)"
            ),
            author="Feedo",
            hooks=[PluginHook.ON_INIT],
        )

    def on_init(self, context: dict[str, Any]) -> None:
        try:
            from praisonaiagents.memory.adapters import register_memory_factory
        except ImportError as exc:
            # Only swallow a genuinely-absent root package. Anything else
            # (broken transitive import, or an incompatible praisonaiagents
            # without `register_memory_factory`) must surface, not be masked
            # as "optional dependency not installed".
            if exc.name in (
                "praisonaiagents",
                "praisonaiagents.memory",
                "praisonaiagents.memory.adapters",
            ):
                logger.warning(
                    "[INTEGRATION] Feedo memory adapter not available — "
                    "install praisonaiagents with memory support. (%s)",
                    exc,
                )
                return
            raise

        register_memory_factory("feedo", create_feedo_memory_adapter)
        logger.info("[INTEGRATION] Feedo memory adapter registered.")
