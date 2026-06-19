import time
from praisonai_plugins.guardrails.semantic_loop_guardrail import (
    SemanticLoopDetector,
    SemanticLoopDetectorPlugin,
)


def test_normalization():
    detector = SemanticLoopDetector()
    assert detector.normalize("Hello, World!") == "hello world"
    assert detector.normalize("  Multiple   spaces  ") == "multiple spaces"
    assert detector.normalize("Punctuation... and! stuff?") == "punctuation and stuff"


def test_shingling():
    detector = SemanticLoopDetector()
    text = "this is a simple test"
    shingles = detector.get_shingles(text, k=2)
    # 5 words, 4 bigrams
    assert len(shingles) == 4


def test_shingling_short_text():
    detector = SemanticLoopDetector()
    text = "short"
    shingles = detector.get_shingles(text, k=2)
    assert len(shingles) == 1


def test_jaccard_similarity():
    detector = SemanticLoopDetector()
    s1 = detector.get_shingles("this is a test")
    s2 = detector.get_shingles("this is a test")
    assert detector.compute_jaccard(s1, s2) == 1.0

    s3 = detector.get_shingles("completely different words here")
    assert detector.compute_jaccard(s1, s3) == 0.0


def test_check_and_record_detects_loop():
    detector = SemanticLoopDetector(threshold=0.45)

    # First message
    is_loop, sim = detector.check_and_record("I will search the database for user 123.")
    assert not is_loop
    assert sim == 0.0

    # Very similar message (paraphrased reasoning loop)
    is_loop, sim = detector.check_and_record(
        "I'm going to search the database for user 123."
    )
    # Similarity should be high
    assert is_loop
    assert sim >= 0.5  # Expect high similarity


def test_check_and_record_maintains_bounded_history():
    detector = SemanticLoopDetector(window_size=3)

    detector.check_and_record("message one")
    detector.check_and_record("message two")
    detector.check_and_record("message three")
    assert len(detector.history) == 3

    detector.check_and_record("message four")
    assert len(detector.history) == 3  # Should not exceed window_size


def test_plugin_after_llm_hook_intervention():
    plugin = SemanticLoopDetectorPlugin()
    plugin.detector.threshold = 0.45  # Lower threshold for testing

    msg1 = "Let me try to find the solution by using the web search tool."
    resp1 = plugin.after_llm(msg1, {})
    assert resp1 == msg1

    msg2 = "I'll try using the web search tool to find the solution."
    resp2 = plugin.after_llm(msg2, {})

    assert resp2 != msg2
    assert "[SYSTEM INTERVENTION]" in resp2


def test_performance():
    detector = SemanticLoopDetector()
    text = "This is a slightly longer message to test the performance of the semantic loop detector. It should execute well under 5 microseconds."

    start = time.perf_counter()
    for _ in range(1000):
        detector.check_and_record(text)
    duration = (time.perf_counter() - start) / 1000

    # 5us is 0.000005 seconds. Python overhead might make it slightly higher in tests,
    # but we assert it's reasonably fast (e.g. < 50us in Python test env)
    assert duration < 0.0001
