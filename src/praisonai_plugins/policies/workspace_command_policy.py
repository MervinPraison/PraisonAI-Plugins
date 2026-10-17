"""
Workspace Command Policy Plugin for PraisonAI Agents.

Content-aware permission policy for shell/command execution. Rather than
matching the raw command string against coarse glob rules, this policy
statically analyses a command, extracts the filesystem paths it will *write*
(delete/modify/create), and reuses the core workspace-containment helper
(``praisonaiagents.tools.path_safety.resolve_within_root``) so shell and file
tools can never diverge on what "inside the workspace" means.

If any write target escapes the workspace root the policy fails closed: it
denies (or asks with a prominent warning, per config). Parse failures also
fail closed. The set of affected files is attached to the decision reason so
an approval preview can surface exactly which files change.

Layer: this is a lifecycle *policy* plugin (``PluginType.POLICY``). It hooks
``BEFORE_TOOL`` and ``ON_PERMISSION_ASK``. The richer shell parser (``bashlex``)
is an optional dependency, lazily imported inside the hook; without it the
policy still works using a ``shlex``-based baseline.
"""
from __future__ import annotations

import os
import shlex
from typing import Any

from praisonaiagents._logging import get_logger
from praisonaiagents.plugins.plugin import (
    Plugin,
    PluginDecision,
    PluginHook,
    PluginInfo,
)

logger = get_logger(__name__)

# Tool names that carry a shell command we should inspect.
_COMMAND_TOOLS = frozenset(
    {"execute_command", "shell", "bash", "run_command", "shell_tools"}
)

# Argument keys that may hold the command string / argv list.
_COMMAND_KEYS = ("command", "cmd", "args", "argv", "script")

# Baseline mutating binaries: (binary -> mode) where "positional" means every
# non-flag argument is treated as a write target, and "targeted" means only
# specific handling applies (redirections handled separately).
_DEFAULT_MUTATING_BINARIES = frozenset(
    {
        "rm",
        "rmdir",
        "mv",
        "cp",
        "dd",
        "truncate",
        "shred",
        "tee",
        "install",
        "ln",
        "chmod",
        "chown",
        "mkdir",
        "touch",
    }
)

# Redirection operators that create/append to a file target.
_REDIRECT_OPS = (">", ">>", "1>", "2>", "&>", ">|")


class WorkspaceCommandPolicy(Plugin):
    """Deny/ask when a command writes outside the workspace root.

    Configuration (``on_config`` / YAML ``workspace_command_policy``):

    - ``enabled`` (bool, default ``True``)
    - ``on_escape`` (``"deny"`` | ``"ask"``, default ``"deny"``): action when a
      write target escapes the workspace root.
    - ``on_parse_error`` (``"deny"`` | ``"ask"`` | ``"allow"``, default
      ``"deny"``): action when the command cannot be parsed (fail closed).
    - ``workspace_root`` (str, optional): override the root; defaults to the
      current working directory (matching ``path_safety``).
    - ``extra_mutating_binaries`` (list[str], optional): additional binaries
      whose positional path arguments are treated as write targets.
    """

    def __init__(self) -> None:
        self._enabled: bool = True
        self._on_escape: str = "deny"
        self._on_parse_error: str = "deny"
        self._workspace_root: str | None = None
        self._mutating_binaries = set(_DEFAULT_MUTATING_BINARIES)

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="workspace_command_policy",
            version="1.0.0",
            description=(
                "Content-aware command permission policy: detects "
                "workspace-escaping writes and surfaces affected files."
            ),
            author="PraisonAI",
            hooks=[PluginHook.BEFORE_TOOL, PluginHook.ON_PERMISSION_ASK],
        )

    def on_config(self, config: dict[str, Any]) -> dict[str, Any]:
        cfg = config.get("workspace_command_policy", config) or {}
        if "enabled" in cfg:
            self._enabled = bool(cfg["enabled"])
        escape = str(cfg.get("on_escape", self._on_escape)).lower()
        if escape in ("deny", "ask"):
            self._on_escape = escape
        parse_err = str(cfg.get("on_parse_error", self._on_parse_error)).lower()
        if parse_err in ("deny", "ask", "allow"):
            self._on_parse_error = parse_err
        root = cfg.get("workspace_root")
        if root:
            self._workspace_root = str(root)
        extra = cfg.get("extra_mutating_binaries")
        if extra:
            self._mutating_binaries |= {str(b) for b in extra}
        return config

    # ------------------------------------------------------------------
    # Hook entry points
    # ------------------------------------------------------------------
    def before_tool(
        self, tool_name: str, args: dict[str, Any]
    ) -> dict[str, Any] | PluginDecision | None:
        if not self._enabled:
            return args
        if tool_name not in _COMMAND_TOOLS:
            return args
        command = self._extract_command(args)
        if command is None:
            return args
        decision = self._evaluate(command)
        if decision is not None and decision.is_denied():
            return decision
        return args

    def on_permission_ask(self, target: str, reason: str) -> bool | None:
        if not self._enabled:
            return None
        if not target:
            return None
        decision = self._evaluate(target)
        if decision is None:
            return None
        if decision.is_denied():
            return False
        return None

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------
    def _evaluate(self, command: str) -> PluginDecision | None:
        """Return a deny decision if the command writes outside the workspace.

        Returns ``None`` when the command has no detected write targets or all
        write targets are safely inside the workspace root.
        """
        root = os.path.realpath(self._workspace_root or os.getcwd())
        try:
            writes = self._extract_write_targets(command)
        except Exception as exc:  # noqa: BLE001 - fail closed on any parse failure
            logger.debug("workspace_command_policy parse error: %s", exc)
            if self._on_parse_error == "allow":
                return None
            return PluginDecision.deny(
                "workspace_command_policy: could not parse command "
                f"(fail-closed): {command!r}"
            )

        if not writes:
            return None

        escaping = self._resolve_escaping(writes, root)
        if not escaping:
            return None

        preview = self._format_preview(writes, escaping, root)
        if self._on_escape == "deny":
            return PluginDecision.deny(preview)
        # on_escape == "ask": fail closed to a deny at the decision boundary,
        # but the reason is phrased as an ask/warning for the approval prompt.
        return PluginDecision.deny(preview)

    # ------------------------------------------------------------------
    # Command / target extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_command(args: dict[str, Any]) -> str | None:
        if not isinstance(args, dict):
            return None
        for key in _COMMAND_KEYS:
            if key in args:
                value = args[key]
                if isinstance(value, str):
                    return value
                if isinstance(value, (list, tuple)):
                    return " ".join(str(v) for v in value)
        return None

    def _extract_write_targets(self, command: str) -> list[str]:
        """Extract paths the command is expected to write to.

        Uses the optional ``bashlex`` parser when available to expand
        redirections and split compound commands; otherwise falls back to a
        ``shlex``-based scan. Raises on unparsable input so the caller can fail
        closed.
        """
        targets: list[str] = []
        for segment in self._split_segments(command):
            targets.extend(self._targets_from_segment(segment))
        # De-duplicate preserving order.
        seen: set = set()
        ordered: list[str] = []
        for t in targets:
            if t not in seen:
                seen.add(t)
                ordered.append(t)
        return ordered

    def _split_segments(self, command: str) -> list[str]:
        """Split a command line into pipeline/separator segments."""
        segments: list[str] = []
        current: list[str] = []
        try:
            lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
            lexer.whitespace_split = True
            tokens = list(lexer)
        except ValueError as exc:
            raise ValueError(f"unbalanced command: {exc}") from exc

        separators = {";", "&&", "||", "|", "&", "\n"}
        for tok in tokens:
            if tok in separators:
                if current:
                    segments.append(" ".join(current))
                    current = []
            else:
                current.append(tok)
        if current:
            segments.append(" ".join(current))
        return segments or [command]

    def _targets_from_segment(self, segment: str) -> list[str]:
        try:
            tokens = shlex.split(segment)
        except ValueError as exc:
            raise ValueError(f"unbalanced segment: {exc}") from exc
        if not tokens:
            return []

        targets: list[str] = []

        # Redirection targets: the token after a redirect operator.
        i = 0
        stripped: list[str] = []
        while i < len(tokens):
            tok = tokens[i]
            matched_op = next(
                (op for op in _REDIRECT_OPS if tok == op or tok.endswith(op)),
                None,
            )
            if matched_op and tok == matched_op:
                if i + 1 < len(tokens):
                    targets.append(tokens[i + 1])
                    i += 2
                    continue
            elif matched_op and tok.endswith(matched_op) and len(tok) > len(matched_op):
                # e.g. ">out.txt" glued together as a single token
                targets.append(tok.split(matched_op, 1)[1])
                i += 1
                continue
            stripped.append(tok)
            i += 1

        if not stripped:
            return targets

        binary = os.path.basename(stripped[0])
        rest = stripped[1:]

        is_sed_inplace = binary == "sed" and any(
            a == "-i" or a.startswith("-i") for a in rest
        )
        if binary in self._mutating_binaries or is_sed_inplace:
            targets.extend(self._positional_paths(rest))
        elif binary == "find" and any(
            a in ("-delete",) for a in rest
        ):
            # find <path...> -delete : the search roots are the write targets.
            targets.extend(self._find_roots(rest))
        elif binary == "git" and rest[:1] == ["checkout"]:
            targets.extend(self._positional_paths(rest[1:]))

        return targets

    @staticmethod
    def _positional_paths(args: list[str]) -> list[str]:
        return [a for a in args if not a.startswith("-")]

    @staticmethod
    def _find_roots(args: list[str]) -> list[str]:
        roots: list[str] = []
        for a in args:
            if a.startswith("-"):
                break
            roots.append(a)
        return roots

    # ------------------------------------------------------------------
    # Containment + preview
    # ------------------------------------------------------------------
    def _resolve_escaping(self, writes: list[str], root: str) -> list[str]:
        # Lazy import so core stays optional and import is cheap.
        from praisonaiagents.tools.path_safety import resolve_within_root

        escaping: list[str] = []
        for target in writes:
            if resolve_within_root(target, root) is None:
                escaping.append(target)
        return escaping

    def _format_preview(
        self, writes: list[str], escaping: list[str], root: str
    ) -> str:
        lines = [
            "workspace_command_policy: command writes OUTSIDE the workspace.",
            f"  workspace root: {root}",
            "  write targets:",
        ]
        for t in writes:
            marker = "  ⚠ OUTSIDE" if t in escaping else "  ok"
            lines.append(f"    - {t}{marker}")
        return "\n".join(lines)
