"""
Follow-Through Plugin for PraisonAI Agents.

Turns the gateway from a purely reactive request/response bot into a genuinely
*proactive* one by closing the open loops that arise naturally in conversation.

After a turn completes (``AFTER_AGENT``), this plugin performs a single, bounded
extraction pass over the transcript to find **commitments**:

  * **agent promises** -- e.g. "I'll check back after your interview tomorrow",
  * **inferred user-context open-loops** -- e.g. "my deadline is Friday".

Each candidate carries a ``kind``, a ``due-window``, a ``confidence`` and a
``dedupe`` key. Candidates above the confidence threshold are persisted and a
*single* proactive check-in is scheduled per commitment via the core scheduler
and ``DeliveryTarget``, routed back to the originating channel/session. A
per-commitment dedupe key guarantees the same loop is never delivered twice.

Design guarantees (safe by default, off unless enabled):

  * **Never mutates the turn** -- ``after_agent`` always returns ``response``
    unchanged; extraction runs strictly as a best-effort side-effect.
  * **Bounded cost** -- at most one LLM pass per turn (heuristic fallback if no
    LLM is available), a confidence threshold, and a cap on pending commitments.
  * **Operator switch** -- disabled unless ``PRAISONAI_FOLLOW_THROUGH`` is truthy
    (or enabled via ``on_config``); every optional dependency is lazy-imported.
  * **Failure isolation** -- any extraction/scheduling error is swallowed and
    logged; it can never affect the main response path.

Lifecycle per commitment: ``pending -> sent | dismissed | snoozed | expired``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from praisonaiagents._logging import get_logger
from praisonaiagents.plugins.plugin import Plugin, PluginHook, PluginInfo

logger = get_logger(__name__)


_ENABLE_ENV = "PRAISONAI_FOLLOW_THROUGH"
_THRESHOLD_ENV = "PRAISONAI_FOLLOW_THROUGH_THRESHOLD"
_MAX_PENDING_ENV = "PRAISONAI_FOLLOW_THROUGH_MAX_PENDING"

_DEFAULT_THRESHOLD = 0.6
_DEFAULT_MAX_PENDING = 100

# Rough "how far out" hints -> seconds. Best-effort; the scheduler owns exact
# timing. These keep the heuristic path dependency-free.
_ONE_HOUR = 3600
_ONE_DAY = 24 * _ONE_HOUR


class CommitmentKind(str, Enum):
    """The kind of open loop a commitment represents."""

    AGENT_PROMISE = "agent_promise"
    EVENT_CHECK_IN = "event_check_in"
    DEADLINE = "deadline"


class CommitmentStatus(str, Enum):
    """Lifecycle status of a persisted commitment."""

    PENDING = "pending"
    SENT = "sent"
    DISMISSED = "dismissed"
    SNOOZED = "snoozed"
    EXPIRED = "expired"


@dataclass
class Commitment:
    """A single extracted open loop scheduled for a proactive check-in."""

    kind: CommitmentKind
    summary: str
    dedupe_key: str
    due_at: float
    confidence: float
    channel: str | None = None
    session_id: str | None = None
    status: CommitmentStatus = CommitmentStatus.PENDING
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["status"] = self.status.value
        return data


class CommitmentStore:
    """A tiny in-memory, thread-safe commitment store with dedupe + a cap.

    Intentionally minimal (no third-party deps). ``add_if_new`` is the dedupe
    gate: a commitment whose ``dedupe_key`` is already tracked -- in any status
    other than a re-openable one -- is rejected so the same loop is never
    scheduled twice. A ``max_pending`` cap bounds unbounded growth.
    """

    def __init__(self, max_pending: int = _DEFAULT_MAX_PENDING) -> None:
        self._lock = threading.Lock()
        self._by_key: dict[str, Commitment] = {}
        self._max_pending = max(1, int(max_pending))

    def add_if_new(self, commitment: Commitment) -> bool:
        """Persist ``commitment`` iff its dedupe key is unseen and cap allows.

        Returns ``True`` if it was newly stored, ``False`` otherwise.
        """
        with self._lock:
            if commitment.dedupe_key in self._by_key:
                return False
            pending = sum(
                1
                for c in self._by_key.values()
                if c.status == CommitmentStatus.PENDING
            )
            if pending >= self._max_pending:
                logger.warning(
                    "[FOLLOW-THROUGH] pending cap reached (%d); dropping %r",
                    self._max_pending,
                    commitment.dedupe_key,
                )
                return False
            self._by_key[commitment.dedupe_key] = commitment
            return True

    def get(self, dedupe_key: str) -> Commitment | None:
        with self._lock:
            return self._by_key.get(dedupe_key)

    def list(
        self, status: CommitmentStatus | None = None
    ) -> list[Commitment]:
        with self._lock:
            values = list(self._by_key.values())
        if status is None:
            return values
        return [c for c in values if c.status == status]

    def set_status(
        self, dedupe_key: str, status: CommitmentStatus
    ) -> bool:
        with self._lock:
            commitment = self._by_key.get(dedupe_key)
            if commitment is None:
                return False
            commitment.status = status
            return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_key)


def _dedupe_key(*parts: str) -> str:
    """Stable short key from normalised parts (channel/session + topic)."""
    joined = "\x1f".join(p.strip().lower() for p in parts if p)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


# Heuristic phrase patterns used when no LLM is available. Deliberately
# conservative: only clear, common commitment/open-loop signals are matched so
# the dependency-free fallback stays high-precision.
_AGENT_PROMISE_RE = re.compile(
    r"\bI(?:'ll| will| shall)\b[^.!?\n]*?\b"
    r"(?:follow up|follow-up|check (?:back|in)|circle back|get back to you|"
    r"remind you|touch base)\b",
    re.IGNORECASE,
)
_EVENT_RE = re.compile(
    r"\b(?:my |the |your )?(interview|meeting|appointment|exam|flight|"
    r"presentation|demo|call|deadline|due date)\b",
    re.IGNORECASE,
)
_WHEN_RE = re.compile(
    r"\b(today|tonight|tomorrow|next week|this week|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"in an hour|in a few hours|later)\b",
    re.IGNORECASE,
)


def _when_to_seconds(when: str | None) -> int:
    """Map a coarse 'when' phrase to a delay in seconds (best-effort)."""
    if not when:
        return _ONE_DAY
    w = when.lower()
    if w in ("today", "tonight", "later"):
        return 6 * _ONE_HOUR
    if w == "in an hour":
        return _ONE_HOUR
    if w == "in a few hours":
        return 3 * _ONE_HOUR
    if w == "tomorrow":
        return _ONE_DAY
    if w in ("this week",):
        return 3 * _ONE_DAY
    if w in ("next week",):
        return 7 * _ONE_DAY
    # A specific weekday -- treat as "within a few days" without a calendar dep.
    return 2 * _ONE_DAY


def _heuristic_extract(
    transcript: str, response: str
) -> list[dict[str, Any]]:
    """Dependency-free, high-precision extraction fallback.

    Returns raw candidate dicts (``kind``/``summary``/``when``/``confidence``/
    ``topic``) which the caller normalises into :class:`Commitment` objects.
    """
    candidates: list[dict[str, Any]] = []

    # 1) Agent promise -- look in the agent's own response.
    if _AGENT_PROMISE_RE.search(response or ""):
        when_match = _WHEN_RE.search(response or "")
        event_match = _EVENT_RE.search(response or "")
        topic = (event_match.group(1) if event_match else "follow_up").lower()
        candidates.append(
            {
                "kind": CommitmentKind.AGENT_PROMISE,
                "summary": (response or "").strip()[:200],
                "when": when_match.group(1) if when_match else None,
                "confidence": 0.7,
                "topic": topic,
            }
        )

    # 2) User-context open loop -- an event/deadline mentioned in the transcript.
    event_match = _EVENT_RE.search(transcript or "")
    if event_match:
        when_match = _WHEN_RE.search(transcript or "")
        topic = event_match.group(1).lower()
        kind = (
            CommitmentKind.DEADLINE
            if topic in ("deadline", "due date")
            else CommitmentKind.EVENT_CHECK_IN
        )
        candidates.append(
            {
                "kind": kind,
                "summary": event_match.group(0).strip()[:200],
                "when": when_match.group(1) if when_match else None,
                "confidence": 0.65,
                "topic": topic,
            }
        )

    return candidates


_LLM_PROMPT = (
    "You extract follow-up commitments from a conversation so an assistant can "
    "proactively close its own open loops. Return ONLY a JSON array (no prose). "
    "Each item: {\"kind\": one of [\"agent_promise\",\"event_check_in\","
    "\"deadline\"], \"summary\": short string, \"when\": coarse timing phrase or "
    "null, \"confidence\": 0..1, \"topic\": one-or-two word dedupe topic}. "
    "Only include CLEAR commitments (an agent promise to follow up, or a user "
    "event/deadline worth checking in on). Return [] if none.\n\n"
    "TRANSCRIPT:\n{transcript}\n\nAGENT_RESPONSE:\n{response}\n\nJSON:"
)


def _llm_extract(
    transcript: str, response: str
) -> list[dict[str, Any]] | None:
    """Single bounded LLM extraction pass. Returns ``None`` if unavailable.

    The LLM client is lazy-imported so the plugin has no hard dependency on any
    model backend; any failure returns ``None`` and the caller falls back to the
    heuristic extractor.
    """
    try:
        from praisonaiagents.llm import LLM  # lazy, optional
    except Exception:  # noqa: BLE001 - optional backend; fall back to heuristic
        return None

    prompt = _LLM_PROMPT.replace("{transcript}", transcript or "").replace(
        "{response}", response or ""
    )
    try:
        model = os.environ.get("PRAISONAI_FOLLOW_THROUGH_MODEL")
        client = LLM(model=model) if model else LLM()
        raw = client.get_response(prompt=prompt, temperature=0.0)
    except Exception as exc:  # noqa: BLE001 # pragma: no cover - backend-specific
        logger.debug("[FOLLOW-THROUGH] LLM extraction failed: %s", exc)
        return None

    return _parse_llm_json(raw)


def _parse_llm_json(raw: Any) -> list[dict[str, Any]] | None:
    """Best-effort parse of the model's JSON array output."""
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except Exception:  # noqa: BLE001 - malformed model output; treat as none
        return None
    if not isinstance(data, list):
        return None
    out: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            out.append(item)
    return out


class FollowThroughPlugin(Plugin):
    """Proactively schedule check-ins for conversational open loops.

    Enabled via the ``PRAISONAI_FOLLOW_THROUGH`` env switch or ``on_config``.
    Reuses the core scheduler + ``DeliveryTarget`` for delivery; core exposes
    everything this plugin needs, so no scheduling logic is re-implemented here.
    """

    def __init__(self) -> None:
        self._enabled = _env_truthy(os.environ.get(_ENABLE_ENV, ""))
        self._threshold = _env_float(
            os.environ.get(_THRESHOLD_ENV, ""), _DEFAULT_THRESHOLD
        )
        max_pending = _env_int(
            os.environ.get(_MAX_PENDING_ENV, ""), _DEFAULT_MAX_PENDING
        )
        self.store = CommitmentStore(max_pending=max_pending)

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="follow_through",
            version="1.0.0",
            description=(
                "Extracts agent promises and conversational open-loops after a "
                "turn and schedules a single proactive check-in per commitment."
            ),
            author="PraisonAI",
            hooks=[PluginHook.AFTER_AGENT],
        )

    def on_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Apply operator config (``enabled``/``threshold``/``max_pending``)."""
        if not isinstance(config, dict):
            return config
        if "enabled" in config:
            self._enabled = bool(config["enabled"])
        if "threshold" in config:
            self._threshold = _env_float(
                str(config["threshold"]), self._threshold
            )
        if "max_pending" in config:
            self.store = CommitmentStore(
                max_pending=_env_int(
                    str(config["max_pending"]), _DEFAULT_MAX_PENDING
                )
            )
        return config

    # ------------------------------------------------------------------ hook

    def after_agent(self, response: str, context: dict[str, Any]) -> str:
        """Best-effort post-turn extraction + scheduling. Never mutates output.

        This method *always* returns ``response`` unchanged. All work is wrapped
        so that no extraction/scheduling error can ever affect the main turn.
        """
        if not self._enabled:
            return response
        try:
            self._process_turn(response, context or {})
        except Exception as exc:  # noqa: BLE001 - never let this affect the response
            logger.debug("[FOLLOW-THROUGH] post-turn processing failed: %s", exc)
        return response

    # -------------------------------------------------------------- internals

    def _process_turn(self, response: str, context: dict[str, Any]) -> None:
        transcript = _extract_transcript(context, response)
        channel = context.get("channel")
        session_id = context.get("session_id") or context.get("session")

        raw = _llm_extract(transcript, response)
        if raw is None:
            raw = _heuristic_extract(transcript, response)

        for candidate in raw:
            commitment = self._normalise(candidate, channel, session_id)
            if commitment is None:
                continue
            if commitment.confidence < self._threshold:
                continue
            if self.store.add_if_new(commitment):
                self._schedule_followup(commitment, context)

    def _normalise(
        self,
        candidate: dict[str, Any],
        channel: str | None,
        session_id: str | None,
    ) -> Commitment | None:
        try:
            kind = _coerce_kind(candidate.get("kind"))
            summary = str(candidate.get("summary") or "").strip()
            if not summary:
                return None
            confidence = _clamp01(candidate.get("confidence"))
            when = candidate.get("when")
            topic = str(candidate.get("topic") or summary)[:64]
            due_at = time.time() + _when_to_seconds(when)
            key = _dedupe_key(channel or "", session_id or "", kind.value, topic)
            return Commitment(
                kind=kind,
                summary=summary,
                dedupe_key=key,
                due_at=due_at,
                confidence=confidence,
                channel=channel,
                session_id=session_id,
            )
        except Exception as exc:  # noqa: BLE001 - best-effort; skip bad candidate
            logger.debug("[FOLLOW-THROUGH] normalise failed: %s", exc)
            return None

    def _schedule_followup(
        self, commitment: Commitment, context: dict[str, Any]
    ) -> None:
        """Schedule a single proactive check-in via the core scheduler.

        Both the scheduler and ``DeliveryTarget`` are lazy-imported: if the
        installed core lacks them, we still persist the commitment (so a CLI can
        surface it) but skip scheduling rather than raising.
        """
        try:
            from praisonaiagents.scheduler.models import DeliveryTarget
        except Exception:  # noqa: BLE001 - scheduler optional; persist without it
            logger.debug(
                "[FOLLOW-THROUGH] scheduler unavailable; persisted but not scheduled"
            )
            return

        try:
            target = DeliveryTarget(
                channel=commitment.channel,
                session_id=commitment.session_id,
            )
        except Exception:  # noqa: BLE001 - tolerate differing DeliveryTarget signatures
            # Older/newer signatures: fall back to a minimal construction.
            try:
                target = DeliveryTarget(channel=commitment.channel)
            except Exception:  # noqa: BLE001 - last-resort fallback
                target = None

        message = _followup_message(commitment)
        runner = context.get("scheduler") or _get_scheduler()
        if runner is None:
            logger.info(
                "[FOLLOW-THROUGH] queued (no runner) key=%s due=%.0f: %s",
                commitment.dedupe_key,
                commitment.due_at,
                message,
            )
            return

        try:
            schedule = getattr(runner, "schedule_once", None) or getattr(
                runner, "schedule", None
            )
            if schedule is None:
                raise AttributeError("scheduler has no schedule/schedule_once")
            schedule(
                run_at=commitment.due_at,
                message=message,
                target=target,
                dedupe_key=commitment.dedupe_key,
            )
            self.store.set_status(
                commitment.dedupe_key, CommitmentStatus.SENT
            )
            logger.info(
                "[FOLLOW-THROUGH] scheduled key=%s kind=%s due=%.0f",
                commitment.dedupe_key,
                commitment.kind.value,
                commitment.due_at,
            )
        except Exception as exc:  # noqa: BLE001 - scheduling is best-effort
            logger.debug("[FOLLOW-THROUGH] scheduling failed: %s", exc)


def _followup_message(commitment: Commitment) -> str:
    """Compose the proactive check-in text for a commitment."""
    topic = commitment.summary
    if commitment.kind == CommitmentKind.EVENT_CHECK_IN:
        return f"Just checking in \u2014 how did it go? ({topic})"
    if commitment.kind == CommitmentKind.DEADLINE:
        return f"Following up on your deadline \u2014 how's it going? ({topic})"
    return f"Following up as promised. ({topic})"


def _extract_transcript(context: dict[str, Any], response: str) -> str:
    """Assemble a plain-text transcript from whatever the context provides."""
    if context.get("transcript"):
        transcript = context["transcript"]
        if isinstance(transcript, str):
            return transcript
        if isinstance(transcript, list):
            parts: list[str] = []
            for turn in transcript:
                if isinstance(turn, dict):
                    role = turn.get("role", "")
                    content = turn.get("content", "")
                    parts.append(f"{role}: {content}".strip())
                else:
                    parts.append(str(turn))
            return "\n".join(parts)
    prompt = context.get("prompt") or context.get("message") or ""
    return f"user: {prompt}\nassistant: {response}".strip()


def _get_scheduler() -> Any | None:
    """Best-effort discovery of a shared scheduler runner from core."""
    try:
        from praisonaiagents.scheduler import get_scheduler  # type: ignore

        return get_scheduler()
    except Exception:  # noqa: BLE001 - scheduler discovery is optional
        return None


def _coerce_kind(value: Any) -> CommitmentKind:
    if isinstance(value, CommitmentKind):
        return value
    text = str(value or "").strip().lower()
    for kind in CommitmentKind:
        if kind.value == text:
            return kind
    return CommitmentKind.AGENT_PROMISE


def _clamp01(value: Any) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, num))


def _env_truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _env_int(value: str, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def create_plugin() -> FollowThroughPlugin:
    """Factory used by directory/module loaders."""
    return FollowThroughPlugin()
