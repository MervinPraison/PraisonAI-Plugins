"""Ingress-time link/URL understanding for gateway/bot channels.

Gateway bots already enrich inbound *media* before an agent runs (voice notes
are transcribed, images/albums are handled), but a shared *web link* passes
through as raw text -- the model either guesses at content it never read, must
spend an extra tool round-trip if a web-fetch tool happens to be enabled, or
silently ignores the link when no web tool is wired for that channel.

This plugin closes that parity gap at the ``MESSAGE_RECEIVED`` lifecycle point:
before the agent turn it detects URLs in the inbound text, fetches and briefly
understands each (title + concise summary), and injects that understanding into
the message content the agent receives -- the same first-class treatment voice
and images already get.

Design constraints (matching the issue):
  * **Opt-in** -- inert unless ``PRAISONAI_LINK_UNDERSTANDING`` is truthy.
  * **Bounded** -- max links per message, max chars per summary, per-request
    timeout, and an optional domain allowlist.
  * **Fail-soft** -- a fetch/parse failure never blocks the reply; the message
    degrades to the raw URL exactly as today.
  * **Lightweight** -- no new hard dependency. The fetch backend is a lazy,
    best-effort chain (``crawl4ai`` -> ``trafilatura``/``requests`` -> stdlib
    ``urllib``); each import is attempted only when needed and skipped on
    ImportError.
  * **SSRF-guarded** -- every URL is validated with the same
    ``is_safe_http_url`` check the core web tools use before any network call.

Configuration (environment):
  * ``PRAISONAI_LINK_UNDERSTANDING``        truthy to enable (default off)
  * ``PRAISONAI_LINK_MAX_LINKS``            max URLs per message (default 3)
  * ``PRAISONAI_LINK_MAX_CHARS``            max summary chars per link (default 500)
  * ``PRAISONAI_LINK_TIMEOUT``              per-fetch timeout seconds (default 8)
  * ``PRAISONAI_LINK_ALLOW_DOMAINS``        comma-separated allowlist (default: any)
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from praisonaiagents.plugins.plugin import Plugin, PluginInfo, PluginHook
from praisonaiagents._logging import get_logger

logger = get_logger(__name__)

# Matches http(s) URLs; trailing punctuation is trimmed after the match so a
# link at the end of a sentence ("... see https://x.com/y.") is captured cleanly.
_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)
_TRAILING = ".,;:!?)]}>\"'"

_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

_TRUTHY = {"1", "true", "yes", "on"}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(float(raw))
        return value if value > 0 else default
    except ValueError:
        logger.warning("link_understanding: invalid %s=%r; using %d", name, raw, default)
        return default


def _env_domains(name: str) -> List[str]:
    raw = os.environ.get(name, "")
    return [d.strip().lower() for d in raw.split(",") if d.strip()]


def detect_urls(text: str) -> List[str]:
    """Extract deduped http(s) URLs from ``text`` preserving first-seen order."""
    if not text:
        return []
    seen: set = set()
    urls: List[str] = []
    for match in _URL_RE.findall(text):
        url = match.rstrip(_TRAILING)
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _domain_allowed(url: str, allow_domains: List[str]) -> bool:
    if not allow_domains:
        return True
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in allow_domains)


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", text or "")).strip()


class LinkUnderstandingPlugin(Plugin):
    """Detect + summarise shared links at ingress, before the agent turn."""

    def __init__(self) -> None:
        self.enabled = _env_flag("PRAISONAI_LINK_UNDERSTANDING", False)
        self.max_links = _env_int("PRAISONAI_LINK_MAX_LINKS", 3)
        self.max_chars = _env_int("PRAISONAI_LINK_MAX_CHARS", 500)
        self.timeout = _env_int("PRAISONAI_LINK_TIMEOUT", 8)
        self.allow_domains = _env_domains("PRAISONAI_LINK_ALLOW_DOMAINS")

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="link_understanding",
            version="0.1.0",
            description=(
                "Ingress-time URL detection + fetch + summary injected into the "
                "message before the agent turn (opt-in, bounded, fail-soft)."
            ),
            author="PraisonAI",
            hooks=[PluginHook.MESSAGE_RECEIVED],
        )

    # ------------------------------------------------------------------ hook

    def before_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Enrich inbound message content with short understanding of any links.

        Never blocks delivery: any failure degrades to the original content.
        """
        if not self.enabled:
            return None
        try:
            content = message.get("content") or ""
            if not content:
                return None
            urls = [
                u for u in detect_urls(content)
                if self._is_safe(u) and _domain_allowed(u, self.allow_domains)
            ][: self.max_links]
            if not urls:
                return None

            notes = [note for note in (self._understand(u) for u in urls) if note]
            if not notes:
                return None

            enriched = content + "\n\n[Link understanding]\n" + "\n".join(notes)
            message["content"] = enriched
            return message
        except Exception as exc:  # fail-soft: never block the reply
            logger.debug("link_understanding: skipped (%s)", exc)
            return None

    # -------------------------------------------------------------- internals

    @staticmethod
    def _is_safe(url: str) -> bool:
        """SSRF guard reusing the core web-tool check; safe if unavailable-open? No."""
        try:
            from praisonaiagents.tools.url_safety import is_safe_http_url
            return bool(is_safe_http_url(url))
        except Exception:
            # If the safety helper cannot be imported we cannot vouch for the
            # URL, so refuse it rather than risk an SSRF fetch.
            return False

    def _understand(self, url: str) -> Optional[str]:
        """Fetch + summarise one URL. Returns a "- url: title — summary" note."""
        title, body = self._fetch(url)
        if title is None and body is None:
            return None
        summary = _clean(body or "")[: self.max_chars]
        title_c = _clean(title or "")
        if title_c and summary:
            return f"- {url}: {title_c} — {summary}"
        if title_c:
            return f"- {url}: {title_c}"
        if summary:
            return f"- {url}: {summary}"
        return None

    def _fetch(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """Best-effort fetch chain. Returns (title, text); (None, None) on failure."""
        # 1) crawl4ai (already shipped with praisonaiagents when installed).
        try:
            from importlib import util as _util
            if _util.find_spec("crawl4ai") is not None:
                text = self._fetch_crawl4ai(url)
                if text:
                    return None, text
        except Exception:
            pass

        # 2) raw HTTP via requests, else stdlib urllib.
        html = self._fetch_http(url)
        if not html:
            return None, None

        title_match = _TITLE_RE.search(html)
        title = title_match.group(1) if title_match else None

        # Prefer trafilatura for main-content extraction when available;
        # otherwise strip tags from the raw HTML.
        try:
            import trafilatura  # type: ignore
            extracted = trafilatura.extract(html)
            if extracted:
                return title, extracted
        except Exception:
            pass
        return title, html

    def _fetch_crawl4ai(self, url: str) -> Optional[str]:
        import asyncio

        async def _run() -> Optional[str]:
            from praisonaiagents.tools import crawl4ai  # type: ignore
            result = await crawl4ai(url)
            if isinstance(result, dict):
                return result.get("markdown") or result.get("cleaned_html")
            return None

        try:
            return asyncio.run(asyncio.wait_for(_run(), timeout=self.timeout))
        except RuntimeError:
            # Already inside a running loop (e.g. async bot). Skip -> HTTP fallback.
            return None
        except Exception:
            return None

    def _fetch_http(self, url: str) -> Optional[str]:
        headers = {"User-Agent": "PraisonAI-LinkUnderstanding/0.1"}
        try:
            import requests  # type: ignore
            resp = requests.get(url, timeout=self.timeout, headers=headers)
            resp.raise_for_status()
            return resp.text
        except Exception:
            pass
        try:
            import urllib.request
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as fh:  # nosec B310
                charset = fh.headers.get_content_charset() or "utf-8"
                return fh.read(1_000_000).decode(charset, "replace")
        except Exception:
            return None


def create_plugin() -> LinkUnderstandingPlugin:
    """Factory used by directory/module loaders."""
    return LinkUnderstandingPlugin()
