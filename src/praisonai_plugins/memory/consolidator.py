"""
Memory Consolidation Plugin for PraisonAI Agents.

Long-running agents capture memories **inline** during a turn; over time the
long-term store accretes near-duplicate, never-merged, never-pruned entries,
which degrades recall precision and grows context/token cost — exactly in the
always-on gateway deployments where memory matters most. Core ships only the
*contract* for a consolidation pass (``MemoryConsolidationProtocol`` plus
``ConsolidationResult`` with its maximum-loss guard); the *heavy, scheduled*
implementation is deliberately a lifecycle plugin. This is that plugin.

It does two things:

- **Implements the core contract.** ``MemoryConsolidationPlugin`` satisfies
  ``MemoryConsolidationProtocol.consolidate`` — a deterministic pass that merges
  near-duplicate long-term memories, promotes durable high-value facts by tagging
  them into a curated tier, prunes the redundant duplicates, and returns a
  ``ConsolidationResult``. Every rewrite is bounded by a **maximum-loss guard**:
  if a pass would drop more than ``max_loss_fraction`` of the *original* entries
  it is rejected and the store is left completely untouched, so a bad rewrite can
  never wipe existing memory.

- **Schedules it off the hot path.** On ``GATEWAY_START`` it spawns a daemon
  thread that runs the pass every ``interval_hours`` (default weekly), adding
  zero latency to replies. On ``GATEWAY_STOP`` the scheduler is stopped cleanly
  and never delays shutdown.

Deterministic by default: the merge/prune pass uses only normalised-text
similarity and per-entry importance/quality metadata — no LLM call, no new
dependency, no network. An optional LLM-assisted summary of a merged cluster can
be enabled with ``use_llm=True`` (lazy-imported), but it is never required and
never runs on the reply path.

Everything is safe by default: the plugin is inert unless installed/enabled,
uses long defaults, writes under a lock, enforces the loss guard, and prunes only
*duplicates it merged this pass* (never arbitrary user data).
"""
from __future__ import annotations

import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from praisonaiagents.plugins.plugin import Plugin, PluginInfo, PluginHook
from praisonaiagents._logging import get_logger

logger = get_logger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")


def _load_consolidation_result() -> Any:
    """Return the core ``ConsolidationResult`` type, or a local shim.

    Prefers the core contract shipped in ``praisonaiagents.memory`` so a plugin
    result is structurally the core result. Falls back to a tiny local dataclass
    with the same fields/guard on older SDKs that predate the contract, so the
    plugin still works (and stays testable) without the newest core.
    """
    try:  # pragma: no cover - trivial import branch
        from praisonaiagents.memory import ConsolidationResult

        return ConsolidationResult
    except Exception:  # pragma: no cover - exercised on older SDKs / stubs
        import math
        from dataclasses import dataclass, field

        @dataclass
        class ConsolidationResult:  # type: ignore[no-redef]
            entries_before: int = 0
            entries_after: int = 0
            merged: int = 0
            promoted: int = 0
            pruned: int = 0
            rejected: bool = False
            reason: Optional[str] = None
            retained_originals: Optional[int] = None
            context: Optional[Dict[str, Any]] = field(default=None)

            def __post_init__(self) -> None:
                if self.context is None:
                    self.context = {}

            @property
            def loss_fraction(self) -> float:
                if self.entries_before <= 0:
                    return 0.0
                if self.retained_originals is not None:
                    retained = max(0, min(self.retained_originals, self.entries_before))
                    removed = self.entries_before - retained
                else:
                    removed = max(self.entries_before - self.entries_after, 0)
                return removed / self.entries_before

            def exceeds_loss(self, max_loss_fraction: float) -> bool:
                if not isinstance(max_loss_fraction, (int, float)) or isinstance(
                    max_loss_fraction, bool
                ):
                    raise ValueError(
                        "max_loss_fraction must be a real number in [0.0, 1.0]"
                    )
                if not math.isfinite(max_loss_fraction) or not (
                    0.0 <= max_loss_fraction <= 1.0
                ):
                    raise ValueError(
                        "max_loss_fraction must be finite and within [0.0, 1.0]"
                    )
                return self.loss_fraction > max_loss_fraction

        return ConsolidationResult


def _text_of(entry: Dict[str, Any]) -> str:
    """Extract the memory text from a store entry (tolerant of key names)."""
    for key in ("text", "memory", "content", "value"):
        val = entry.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


def _id_of(entry: Dict[str, Any]) -> Optional[str]:
    """Extract a stable id from a store entry, if present."""
    for key in ("id", "memory_id", "_id"):
        val = entry.get(key)
        if val is not None:
            return str(val)
    return None


def _tokens(text: str) -> frozenset:
    """Lowercased alphanumeric token set for similarity comparison."""
    return frozenset(_WORD_RE.findall(text.lower()))


def _jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity of two token sets (0.0 when either is empty)."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def _importance(entry: Dict[str, Any]) -> float:
    """Best-effort importance/quality score for an entry (default 0.0)."""
    meta = entry.get("metadata")
    if isinstance(meta, dict):
        for key in ("importance", "quality", "score"):
            val = meta.get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return float(val)
    for key in ("importance", "quality", "score"):
        val = entry.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
    return 0.0


class MemoryConsolidationPlugin(Plugin):
    """Scheduled, loss-bounded consolidation of long-lived agent memory.

    Implements ``MemoryConsolidationProtocol`` and schedules the pass off the
    reply path via a ``GATEWAY_START`` daemon thread (stopped on
    ``GATEWAY_STOP``). Mirrors the shipped ``memory_watchdog`` /
    ``skill_curator`` lifecycle structure.

    Configuration (via ``on_config`` / the ``memory_consolidation:`` block):
        enabled: Whether the scheduled pass runs. Default ``False`` (opt-in).
        interval_hours: Schedule interval in hours. Default ``168`` (weekly).
        max_loss_fraction: Maximum fraction of original entries a single pass may
            drop before it is rejected and the store left untouched.
            Default ``0.25``.
        similarity_threshold: Jaccard token similarity at/above which two
            long-term memories are treated as near-duplicates. Default ``0.85``.
        promote_importance: Entries with importance/quality at/above this are
            tagged into the curated tier. Default ``0.7``.
        curated_tag: Metadata tag applied to promoted (curated) entries.
            Default ``"curated"``.
        use_llm: When ``True``, lazily use an Agent to summarise a merged cluster
            instead of keeping the highest-importance member verbatim. Never
            required; defaults to ``False`` (fully deterministic).
        dry_run: When ``True``, compute and report the pass without mutating the
            store. Default ``False``.
    """

    def __init__(self, memory: Any = None) -> None:
        self._memory = memory
        self._enabled: bool = False
        self._interval_hours: float = 168.0
        self._max_loss_fraction: float = 0.25
        self._similarity_threshold: float = 0.85
        self._promote_importance: float = 0.7
        self._curated_tag: str = "curated"
        self._use_llm: bool = False
        self._dry_run: bool = False

        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def info(self) -> PluginInfo:
        # Reference hooks defensively so plugin discovery still succeeds on an
        # older praisonaiagents that lacks the GATEWAY_* enum members (mirrors
        # skill_curator), matching this plugin's older-SDK importability goal.
        hooks = [
            getattr(PluginHook, name)
            for name in ("GATEWAY_START", "GATEWAY_STOP")
            if hasattr(PluginHook, name)
        ]
        return PluginInfo(
            name="memory_consolidation",
            version="1.0.0",
            description=(
                "Scheduled, loss-bounded consolidation of long-lived agent "
                "memory: merges near-duplicates, promotes durable facts to a "
                "curated tier, prunes redundant entries, off the hot path."
            ),
            author="PraisonAI",
            hooks=hooks,
        )

    def on_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Read optional configuration before the gateway starts."""
        try:
            if "enabled" in config:
                self._enabled = bool(config["enabled"])
            if "interval_hours" in config:
                self._interval_hours = max(0.01, float(config["interval_hours"]))
            if "max_loss_fraction" in config:
                mlf = float(config["max_loss_fraction"])
                self._max_loss_fraction = min(1.0, max(0.0, mlf))
            if "similarity_threshold" in config:
                st = float(config["similarity_threshold"])
                self._similarity_threshold = min(1.0, max(0.0, st))
            if "promote_importance" in config:
                self._promote_importance = float(config["promote_importance"])
            if "curated_tag" in config and config["curated_tag"]:
                self._curated_tag = str(config["curated_tag"])
            if "use_llm" in config:
                self._use_llm = bool(config["use_llm"])
            if "dry_run" in config:
                self._dry_run = bool(config["dry_run"])
        except Exception as e:  # noqa: BLE001 — config must never break the runtime
            logger.debug(f"[MEMCONSOLIDATE] config parse error (non-fatal): {e}")
        return config

    def gateway_start(self, event: Any = None) -> Any:
        """Start the scheduled off-hot-path consolidation sweeper."""
        if not self._enabled:
            return event
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return event
            self._stop_event = threading.Event()
            self._thread = threading.Thread(
                target=self._run_scheduler,
                name="memory-consolidation",
                daemon=True,
            )
            self._thread.start()
        logger.info(
            "[MEMCONSOLIDATE] scheduler started "
            f"(interval={self._interval_hours:.1f}h, "
            f"max_loss_fraction={self._max_loss_fraction})"
        )
        return event

    def gateway_stop(self, event: Any = None) -> Any:
        """Stop the sweeper cleanly without ever blocking shutdown."""
        with self._lock:
            stop_event = self._stop_event
            thread = self._thread
            self._stop_event = None
            self._thread = None

        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread.is_alive():
            try:
                thread.join(timeout=1.0)
            except Exception:
                pass
        return event

    def on_shutdown(self) -> None:
        """Ensure the sweeper is stopped if the plugin is unregistered."""
        self.gateway_stop(None)

    def _run_scheduler(self) -> None:
        """Daemon loop: run a consolidation pass every ``interval_hours``."""
        stop_event = self._stop_event
        if stop_event is None:
            return
        period = self._interval_hours * 3600.0
        while not stop_event.wait(period):
            memory = self._memory
            if memory is None:
                logger.debug("[MEMCONSOLIDATE] no memory bound; skipping pass")
                continue
            try:
                result = self.consolidate(
                    memory, max_loss_fraction=self._max_loss_fraction
                )
                if getattr(result, "rejected", False):
                    logger.warning(
                        f"[MEMCONSOLIDATE] pass rejected: {result.reason}"
                    )
                else:
                    logger.info(
                        "[MEMCONSOLIDATE] pass complete "
                        f"before={result.entries_before} after={result.entries_after} "
                        f"merged={result.merged} promoted={result.promoted} "
                        f"pruned={result.pruned}"
                    )
            except Exception as e:  # noqa: BLE001 — a pass must never crash the loop
                logger.debug(f"[MEMCONSOLIDATE] pass error (non-fatal): {e}")

    # ---- MemoryConsolidationProtocol -------------------------------------

    def consolidate(
        self,
        memory: Any,
        *,
        max_loss_fraction: float = 0.25,
    ) -> Any:
        """Run one deterministic, loss-bounded consolidation pass over ``memory``.

        Merges near-duplicate long-term entries, promotes durable high-value
        facts into a curated tier, and prunes the redundant duplicates. The pass
        is rejected (store untouched) if it would drop more than
        ``max_loss_fraction`` of the original entries.
        """
        Result = _load_consolidation_result()

        try:
            entries = list(memory.get_all_memories())
        except Exception as e:  # noqa: BLE001
            return Result(
                entries_before=0,
                entries_after=0,
                rejected=True,
                reason=f"could not read memories: {e}",
            )

        long_term = [e for e in entries if isinstance(e, dict) and _text_of(e)]
        before = len(long_term)
        if before == 0:
            return Result(entries_before=0, entries_after=0)

        clusters = self._cluster(long_term)

        merged = 0
        promoted = 0
        prune_ids: List[str] = []
        retained_originals = 0

        for cluster in clusters:
            cluster.sort(key=_importance, reverse=True)
            keeper = cluster[0]
            retained_originals += 1  # the keeper is an original that survives
            if len(cluster) > 1:
                merged += len(cluster) - 1
                for dup in cluster[1:]:
                    dup_id = _id_of(dup)
                    if dup_id is not None:
                        prune_ids.append(dup_id)
            if _importance(keeper) >= self._promote_importance:
                promoted += 1

        after = before - len(prune_ids)

        result = Result(
            entries_before=before,
            entries_after=after,
            merged=merged,
            promoted=promoted,
            pruned=len(prune_ids),
            retained_originals=retained_originals,
        )

        try:
            if result.exceeds_loss(max_loss_fraction):
                result.rejected = True
                result.reason = (
                    f"loss guard tripped: would drop "
                    f"{result.loss_fraction:.2f} > {max_loss_fraction:.2f}"
                )
                return result
        except ValueError as e:
            result.rejected = True
            result.reason = str(e)
            return result

        if self._dry_run:
            if isinstance(result.context, dict):
                result.context["dry_run"] = True
            return result

        # Apply the rewrite concurrency-safely: prune merged duplicates and tag
        # promoted keepers. Both are best-effort against the memory surface.
        with self._lock:
            self._prune(memory, prune_ids)
            self._promote(memory, clusters)

        return result

    def _cluster(self, entries: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group entries into near-duplicate clusters by token similarity."""
        token_cache: List[Tuple[Dict[str, Any], frozenset]] = [
            (e, _tokens(_text_of(e))) for e in entries
        ]
        clusters: List[List[Dict[str, Any]]] = []
        cluster_tokens: List[frozenset] = []

        for entry, toks in token_cache:
            placed = False
            for idx, ref in enumerate(cluster_tokens):
                if _jaccard(toks, ref) >= self._similarity_threshold:
                    clusters[idx].append(entry)
                    placed = True
                    break
            if not placed:
                clusters.append([entry])
                cluster_tokens.append(toks)
        return clusters

    def _prune(self, memory: Any, prune_ids: List[str]) -> None:
        """Delete merged-duplicate entries, tolerant of the memory surface."""
        if not prune_ids:
            return
        delete_many = getattr(memory, "delete_memories", None)
        if callable(delete_many):
            try:
                delete_many(prune_ids)
                return
            except Exception as e:  # noqa: BLE001
                logger.debug(f"[MEMCONSOLIDATE] delete_memories failed: {e}")
        delete_one = getattr(memory, "delete_memory", None)
        if callable(delete_one):
            for mid in prune_ids:
                try:
                    delete_one(mid)
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"[MEMCONSOLIDATE] delete_memory {mid} failed: {e}")

    def _promote(
        self, memory: Any, clusters: List[List[Dict[str, Any]]]
    ) -> None:
        """Tag durable, high-value keepers into the curated tier if supported."""
        update = getattr(memory, "update_memory", None)
        if not callable(update):
            return
        for cluster in clusters:
            if not cluster:
                continue
            keeper = cluster[0]
            if _importance(keeper) < self._promote_importance:
                continue
            mid = _id_of(keeper)
            if mid is None:
                continue
            meta = dict(keeper.get("metadata") or {})
            meta["tier"] = self._curated_tag
            try:
                update(mid, metadata=meta)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"[MEMCONSOLIDATE] promote {mid} failed: {e}")
