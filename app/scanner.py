"""Score one section of text. No network calls and no model download.

Four checks:
1. Empty text fails immediately.
2. Banned phrases (see config.BANNED_AI_PHRASES).
3. Burstiness: how much sentence length jumps around.
4. Optional neural detector, passed in by the caller. This file never loads it.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable

import textstat

from config import AI_LABELS, BANNED_AI_PHRASES, MAX_AI_PROBABILITY, MIN_BURSTINESS
from schemas import ScanResult

Detector = Callable[[str], list[dict[str, object]]]

_SENTENCE_SPLIT = re.compile(r"[.!?]+")


def scan_text(text: str, detector: Detector | None = None) -> ScanResult:
    """Return the pass/fail card for `text`."""
    if not text.strip():
        return ScanResult(
            passed=False,
            ai_probability=0.0 if detector is not None else None,
            burstiness=0.0,
            grade_level=0.0,
            flagged_words=[],
            detector_used=detector is not None,
        )

    flagged_words = _flagged_phrases(text)
    burstiness = _burstiness(text)
    grade_level = _grade_level(text)
    ai_probability = _ai_percent(text, detector) if detector is not None else None

    ai_ok = True if ai_probability is None else ai_probability < (MAX_AI_PROBABILITY * 100)
    passed = (not flagged_words) and burstiness >= MIN_BURSTINESS and ai_ok

    return ScanResult(
        passed=passed,
        ai_probability=None if ai_probability is None else round(ai_probability, 1),
        burstiness=round(burstiness, 2),
        grade_level=round(grade_level, 1),
        flagged_words=flagged_words,
        detector_used=detector is not None,
    )


def _flagged_phrases(text: str) -> list[str]:
    lower_text = text.lower()
    return [phrase for phrase in BANNED_AI_PHRASES if phrase in lower_text]


def _burstiness(text: str) -> float:
    sentences = [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]
    word_counts = [len(sentence.split()) for sentence in sentences]
    if len(word_counts) < 2:
        return 0.0
    return float(statistics.stdev(word_counts))


def _grade_level(text: str) -> float:
    try:
        return float(textstat.flesch_kincaid_grade(text))
    except (ValueError, ZeroDivisionError, TypeError):
        return 0.0


def _ai_percent(text: str, detector: Detector) -> float:
    """Map the detector label to a 0-100 chance the text is AI-written."""
    result = detector(text)[0]
    label = str(result["label"])
    confidence = float(result["score"])  # type: ignore[arg-type]
    probability = confidence if label in AI_LABELS else 1.0 - confidence
    return probability * 100
