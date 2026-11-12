"""Outline JSON from a model, including the markdown-fence case."""

import pytest

from schemas import PaperOutline
from writer import (
    WriterError,
    _chat,
    _lmstudio_reply,
    _parse_outline,
    _reply_text,
    _strip_model_heading,
    as_paragraph,
    downloaded_model_ids,
)


OUTLINE = """
{
  "title": "Why a small team should keep one app",
  "sections": [
    {
      "heading": "The afternoon we lost",
      "structure_type": "personal_story_narrative",
      "core_points": ["twelve services", "three hour deploy"],
      "personal_touch": "Last year my four person team split one app."
    },
    {
      "heading": "What actually broke",
      "structure_type": "problem_and_fix",
      "core_points": ["network hops", "hard logs"],
      "personal_touch": "We could not tell which service failed first."
    }
  ]
}
"""


def test_parse_plain_json() -> None:
    outline = _parse_outline(OUTLINE)
    assert isinstance(outline, PaperOutline)
    assert outline.sections[0].structure_type == "personal_story_narrative"


def test_parse_json_inside_a_fence() -> None:
    outline = _parse_outline("```json\n" + OUTLINE + "\n```")
    assert outline.title.startswith("Why a small team")


def test_reply_uses_content_when_present() -> None:
    class Message:
        content = "The deploy took three hours."
        thinking = "I should mention the deploy."

    assert _reply_text(Message()) == "The deploy took three hours."


def test_reply_falls_back_to_thinking_when_content_is_empty() -> None:
    class Message:
        content = ""
        thinking = '{"title": "Keep one app"}'

    assert _reply_text(Message()).startswith("{")


def test_as_paragraph_joins_broken_lines() -> None:
    raw = "The team compares models\nagainst simple tools. They\nwant a fair test."
    assert as_paragraph(raw) == "The team compares models against simple tools. They want a fair test."


def test_strip_leading_markdown_heading() -> None:
    text = _strip_model_heading("## The afternoon we lost\n\nIt broke. We fixed it.")
    assert text == "It broke. We fixed it."


def test_downloaded_models_skip_embeddings() -> None:
    names = downloaded_model_ids(
        {
            "data": [
                {"id": "writer-q4", "type": "llm"},
                {"id": "embed-small", "type": "embedding"},
                {"id": "picture-q4", "type": "vlm"},
            ]
        }
    )
    assert names == ["writer-q4", "picture-q4"]


def test_lmstudio_reply_uses_content() -> None:
    text, reason = _lmstudio_reply(
        {"choices": [{"finish_reason": "stop", "message": {"content": "One paragraph."}}]}
    )
    assert text == "One paragraph."
    assert reason == "stop"


def test_lmstudio_reply_falls_back_to_reasoning() -> None:
    text, _reason = _lmstudio_reply(
        {"choices": [{"finish_reason": "stop", "message": {"content": "", "reasoning_content": "The draft."}}]}
    )
    assert text == "The draft."


def test_lmstudio_chat_posts_the_downloaded_model(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_post(url: str, payload: dict[str, object], timeout: float = 600) -> object:
        seen["url"] = url
        seen["payload"] = payload
        return {"choices": [{"finish_reason": "stop", "message": {"content": "Kept the facts."}}]}

    monkeypatch.setattr("writer._post_json", fake_post)
    reply = _chat("writer-q4", "Write it.", temperature=0.7, engine="LM Studio")
    assert reply == "Kept the facts."
    assert str(seen["url"]).endswith("/v1/chat/completions")
    payload = seen["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "writer-q4"


def test_bad_json_raises_writer_error() -> None:
    with pytest.raises(WriterError):
        _parse_outline("this is not json")
