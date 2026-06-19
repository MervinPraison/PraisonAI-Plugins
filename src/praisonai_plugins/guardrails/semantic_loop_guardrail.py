"""
Semantic Loop Detection Guardrail Plugin for PraisonAI Agents.

This plugin detects paraphrased reasoning loops by analyzing the semantic 
similarity of recent LLM responses using k-gram shingling and Jaccard similarity.
"""

import re
import threading
from collections import deque
from typing import Dict, Any, Set, Tuple

from praisonaiagents.plugins.plugin import Plugin, PluginInfo, PluginHook
from praisonaiagents._logging import get_logger

logger = get_logger(__name__)


class SemanticLoopDetector:
    """Core semantic detection algorithm with zero external dependencies."""

    def __init__(self, window_size: int = 5, threshold: float = 0.85):
        self._window_size = window_size
        self.threshold = threshold
        self._local = threading.local()
        
        # Precompile regex for fast text normalization
        self._normalize_re = re.compile(r'[^\w\s]', flags=re.UNICODE)
        self._whitespace_re = re.compile(r'\s+')

    @property
    def window_size(self) -> int:
        return self._window_size

    @window_size.setter
    def window_size(self, size: int):
        self._window_size = size
        # Reinitialize deque with new maxlen if size changes
        if hasattr(self._local, "history"):
            old_history = self._local.history
            self._local.history = deque(old_history, maxlen=size)

    @property
    def history(self) -> deque:
        if not hasattr(self._local, "history"):
            self._local.history = deque(maxlen=self.window_size)
        return self._local.history

    def normalize(self, text: str) -> str:
        """Lowercase, remove punctuation, collapse whitespace."""
        text = text.lower()
        text = self._normalize_re.sub('', text)
        return self._whitespace_re.sub(' ', text).strip()

    def get_shingles(self, text: str, k: int = 2) -> Set[str]:
        """Generate k-gram shingles from normalized text using raw strings."""
        words = self.normalize(text).split()
        if len(words) < k:
            return {" ".join(words)} if words else set()
        
        shingles = set()
        for i in range(len(words) - k + 1):
            shingle = " ".join(words[i:i+k])
            shingles.add(shingle)
        return shingles

    def compute_jaccard(self, set_a: Set[str], set_b: Set[str]) -> float:
        """Compute Jaccard similarity between two sets of shingles."""
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union

    def check_and_record(self, text: str) -> Tuple[bool, float]:
        """
        Check if text is semantically similar to recent history.
        Returns (is_loop, max_similarity_score).
        """
        if not text.strip():
            return False, 0.0

        current_shingles = self.get_shingles(text)
        if not current_shingles:
            return False, 0.0

        max_similarity = 0.0

        for past_shingles in self.history:
            similarity = self.compute_jaccard(current_shingles, past_shingles)
            if similarity > max_similarity:
                max_similarity = similarity

        is_loop = max_similarity >= self.threshold
        self.history.append(current_shingles)
        
        return is_loop, max_similarity


class SemanticLoopDetectorPlugin(Plugin):
    """
    Plugin that intercepts LLM responses and guards against semantic stagnation.
    """
    
    def __init__(self):
        super().__init__()
        self.enabled = True
        self.detector = SemanticLoopDetector(window_size=5, threshold=0.85)

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="semantic_loop_detector",
            version="1.0.0",
            description="Detects paraphrased reasoning loops using Jaccard similarity.",
            author="TokenCircuit",
            hooks=[PluginHook.AFTER_LLM]
        )
        
    def after_llm(self, response: str, usage: Dict[str, Any]) -> str:
        if not self.enabled:
            return response
            
        is_loop, similarity = self.detector.check_and_record(response)
        
        if is_loop:
            logger.warning(
                f"[GUARDRAIL] Semantic doom loop detected! "
                f"(Similarity: {similarity:.2f} >= {self.detector.threshold})"
            )
            intervention = (
                "\n\n[SYSTEM INTERVENTION]: You are repeating a paraphrased version "
                "of your previous thoughts or actions. You are stuck in a reasoning loop. "
                "You MUST change your strategy, use a different tool, or ask the user for help."
            )
            return response + intervention
            
        return response
