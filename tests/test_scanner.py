"""Scanner checks that do not need Ollama or the neural model."""

from scanner import scan_text


def test_empty_text_fails() -> None:
    result = scan_text("   ")
    assert result.passed is False
    assert result.ai_probability is None
    assert result.detector_used is False
    assert result.flagged_words == []


def test_banned_phrase_is_flagged_case_insensitive() -> None:
    text = (
        "I tried this last March. It broke. "
        "In conclusion the deploy was slow and the logs were a mess. "
        "We rolled it back the same night after the pager went off."
    )
    result = scan_text(text)
    assert "in conclusion" in result.flagged_words
    assert result.passed is False


def test_burstiness_is_sentence_length_spread() -> None:
    # One word, then twenty words. Sample stdev of [1, 20] is about 13.44.
    text = "Hi. " + ("word " * 20).strip() + "."
    result = scan_text(text)
    assert result.burstiness > 10
    assert result.flagged_words == []


def test_uniform_sentences_fail_burstiness() -> None:
    sentence = "We shipped the patch on Friday"
    text = ". ".join([sentence] * 4) + "."
    result = scan_text(text)
    assert result.burstiness == 0.0
    assert result.passed is False


def test_varied_human_text_passes_without_detector() -> None:
    text = (
        "It broke. "
        "I sat with the four-person team and watched one deploy eat the whole afternoon. "
        "Three hours. "
        "The old app shipped in about ten minutes, so the new split was hard to defend. "
        "Here is why: twelve services meant twelve logs, and nobody knew which one failed first."
    )
    result = scan_text(text)
    assert result.flagged_words == []
    assert result.burstiness >= 4.2
    assert result.passed is True
    assert result.detector_used is False


def test_detector_label_chat_gpt_counts_as_ai() -> None:
    def fake_detector(_text: str) -> list[dict[str, object]]:
        return [{"label": "ChatGPT", "score": 0.91}]

    text = (
        "It broke. "
        "I sat with the four-person team and watched one deploy eat the whole afternoon. "
        "Three hours. "
        "The old app shipped in about ten minutes, so the new split was hard to defend."
    )
    result = scan_text(text, fake_detector)
    assert result.detector_used is True
    assert result.ai_probability == 91.0
    assert result.passed is False


def test_detector_human_label_inverts_score() -> None:
    def fake_detector(_text: str) -> list[dict[str, object]]:
        return [{"label": "Human", "score": 0.8}]

    text = (
        "It broke. "
        "I sat with the four-person team and watched one deploy eat the whole afternoon. "
        "Three hours. "
        "The old app shipped in about ten minutes, so the new split was hard to defend."
    )
    result = scan_text(text, fake_detector)
    # Human at 0.8 confidence -> 20% AI probability.
    assert result.ai_probability == 20.0
    assert result.passed is True
