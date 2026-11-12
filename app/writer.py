"""Talk to a local model, either Ollama or LM Studio.

Three calls, and nothing else:
- build_outline: rough notes -> title and 3-4 section plans
- draft_section: one plan -> the first draft
- rewrite_section: a draft plus its score -> a new draft

Ollama has to be running on this machine (`ollama serve`). Set OLLAMA_HOST if
it is not on http://127.0.0.1:11434.

LM Studio has to have its local server started. Set LMSTUDIO_HOST if it is
not on http://127.0.0.1:1234. The model list is the models already downloaded
there, not a catalog you can pull from this app.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import ollama

from config import (
    BANNED_AI_PHRASES,
    REWRITE_TARGET_AI_PERCENT,
    REWRITE_TARGET_BURSTINESS,
)
from schemas import PaperOutline, ScanResult, SectionPlan

# Flat schema on purpose. Pydantic's model_json_schema() uses $ref links, and
# Ollama's structured-output parser often rejects those.
OUTLINE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "sections": {
            "type": "array",
            "minItems": 3,
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "heading": {"type": "string"},
                    "structure_type": {
                        "type": "string",
                        "enum": [
                            "personal_story_narrative",
                            "problem_and_fix",
                            "short_bullet_breakdown",
                            "real_example_walkthrough",
                        ],
                    },
                    "core_points": {"type": "array", "items": {"type": "string"}},
                    "personal_touch": {"type": "string"},
                },
                "required": ["heading", "structure_type", "core_points", "personal_touch"],
            },
        },
    },
    "required": ["title", "sections"],
}


class WriterError(RuntimeError):
    """The writer was unreachable, returned nothing, or returned invalid JSON."""


ENGINES = ("Ollama", "LM Studio")
LMSTUDIO_HOST = os.environ.get("LMSTUDIO_HOST", "http://127.0.0.1:1234").rstrip("/")


def writer_status(engine: str) -> tuple[bool, list[str], str]:
    """Return (is_up, downloaded_model_names, detail) for the chosen writer."""
    if engine == "LM Studio":
        return lmstudio_status()
    return ollama_status()


def ollama_status() -> tuple[bool, list[str], str]:
    """Return (is_up, installed_model_names, detail)."""
    try:
        listed = ollama.list()
    except Exception as exc:
        return False, [], str(exc)

    names: list[str] = []
    models = getattr(listed, "models", None) or []
    for model in models:
        name = getattr(model, "model", None) or getattr(model, "name", None)
        if name:
            names.append(str(name))
    return True, names, "Ollama is running."


def lmstudio_status() -> tuple[bool, list[str], str]:
    """Return downloaded LM Studio models. Embeddings are left out."""
    try:
        payload = _get_json(f"{LMSTUDIO_HOST}/api/v0/models")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            return False, [], _lmstudio_down(exc)
        try:
            payload = _get_json(f"{LMSTUDIO_HOST}/v1/models")
        except Exception as inner:
            return False, [], _lmstudio_down(inner)
    except Exception as exc:
        return False, [], _lmstudio_down(exc)
    return True, downloaded_model_ids(payload), "LM Studio is running."


def downloaded_model_ids(payload: object) -> list[str]:
    """Names from an LM Studio model list. Skip anything that is not a writer model."""
    if not isinstance(payload, dict):
        return []
    rows = payload.get("data") or payload.get("models") or []
    names: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        kind = str(row.get("type") or "llm").lower()
        if kind not in {"llm", "vlm"}:
            continue
        name = row.get("id") or row.get("model") or row.get("key")
        if name:
            names.append(str(name))
    return names


def _lmstudio_down(exc: Exception) -> str:
    return (
        f"{exc} Start the local server in LM Studio "
        f"(default {LMSTUDIO_HOST}) and download a model there."
    )


def build_outline(
    rough_notes: str,
    experience_notes: str,
    audience: str,
    model_name: str,
    english_level: str,
    engine: str = "Ollama",
) -> PaperOutline:
    prompt = f"""
The user is a non-native English speaker. Read their rough notes and personal experience.
Do not judge grammar. Extract their true meaning and create a 3 to 4 section paper outline for: {audience}.
Assign a DIFFERENT structure_type to each section so the paper does not look uniform.
The finished paper must be written at this English level, so keep headings and points suitable for it:
{english_level}

Rough Idea Notes: {rough_notes}
Personal Experience / Examples: {experience_notes}
""".strip()
    raw = _chat(model_name, prompt, temperature=0.2, schema=OUTLINE_SCHEMA, engine=engine)
    return _parse_outline(raw)


# A requested "line" is about 12 words. The box wraps the paragraph itself.
WORDS_PER_LINE = 12


def as_paragraph(text: str) -> str:
    """One flowing paragraph. No broken lines in the middle of a sentence."""
    return " ".join(text.split())


def write_paragraph(
    rough_notes: str,
    audience: str,
    model_name: str,
    english_level: str,
    line_count: int,
    engine: str = "Ollama",
) -> str:
    """One paragraph from rough notes, then split into `line_count` lines."""
    lines = max(1, line_count)
    minimum = lines * WORDS_PER_LINE
    maximum = lines * (WORDS_PER_LINE + 4)
    prompt = f"""
Turn these rough notes into one paragraph for: {audience}.
English level you must follow: {english_level}
Length is required: {minimum} to {maximum} words. That is {lines} lines.
One paragraph only. No title, no bullets, no heading.

Rough notes:
{rough_notes}

Rules:
- Follow the English level for every sentence.
- Use only the writer's meaning. Do not add facts, products, or numbers they did not give.
- If the notes are short, explain those same points in more sentences until you reach {minimum} words.
- NEVER use these words: {", ".join(BANNED_AI_PHRASES)}.
- Do not end with a summary sentence.
""".strip()
    draft = _strip_model_heading(_chat(model_name, prompt, temperature=0.7, engine=engine))
    if len(draft.split()) < int(minimum * 0.85):
        draft = _lengthen_paragraph(
            draft, audience, model_name, english_level, lines, minimum, maximum, engine=engine
        )
    return as_paragraph(draft)


def _lengthen_paragraph(
    draft: str,
    audience: str,
    model_name: str,
    english_level: str,
    lines: int,
    minimum: int,
    maximum: int,
    engine: str = "Ollama",
) -> str:
    have = len(draft.split())
    prompt = f"""
This paragraph is too short. It has {have} words.
Rewrite it as one paragraph of {minimum} to {maximum} words, about {lines} lines, for: {audience}.
English level you must follow: {english_level}
Keep the same facts. Do not add new products, names, or numbers.
Explain the same points with more sentences until you reach {minimum} words.
NEVER use these words: {", ".join(BANNED_AI_PHRASES)}.
No title. No bullets.

Paragraph:
{draft}
""".strip()
    return _strip_model_heading(_chat(model_name, prompt, temperature=0.7, engine=engine))


def draft_section(
    section: SectionPlan,
    audience: str,
    model_name: str,
    english_level: str,
    engine: str = "Ollama",
) -> str:
    prompt = f"""
Write one section of a paper for: {audience}.
English level you must follow: {english_level}
Section Heading: {section.heading}
Format Style: {section.structure_type}
Key Points: {section.core_points}
Personal Experience to Include: {section.personal_touch}

STRICT WRITING RULES:
- Follow the English level above for every sentence. Do not write above or below that level.
- Include the personal experience naturally using "I" or "We".
- Vary sentence length inside that level.
- NEVER use these words: {", ".join(BANNED_AI_PHRASES)}.
- Never wrap up with a summary paragraph at the end of the section.
""".strip()
    return _strip_model_heading(_chat(model_name, prompt, temperature=0.7, engine=engine))


def rewrite_section(
    current_text: str,
    section: SectionPlan,
    stats: ScanResult,
    audience: str,
    model_name: str,
    english_level: str,
    extra_hint: str = "",
    engine: str = "Ollama",
) -> str:
    hint_line = ""
    if extra_hint.strip():
        hint_line = (
            f"- Add this personal detail from the author (written in rough English): {extra_hint.strip()}"
        )

    ai_score = "n/a" if stats.ai_probability is None else f"{stats.ai_probability}%"
    prompt = f"""
Rewrite this section for {audience} so it sounds like a real person talking.
English level you must follow: {english_level}
Keep the format style: {section.structure_type}.

Current issues to fix:
- AI Detector Score is {ai_score} (Target: under {REWRITE_TARGET_AI_PERCENT}%).
- Sentence variation (burstiness) is {stats.burstiness} (Target: above {REWRITE_TARGET_BURSTINESS}). Vary sentence length inside the English level above.
- Remove these banned phrases immediately: {stats.flagged_words + BANNED_AI_PHRASES}
- Every sentence must stay at the English level above. Do not write above or below it.
{hint_line}

Draft to rewrite:
{current_text}
""".strip()
    return _strip_model_heading(_chat(model_name, prompt, temperature=0.85, engine=engine))


def _chat(
    model_name: str,
    prompt: str,
    temperature: float,
    schema: dict[str, object] | None = None,
    engine: str = "Ollama",
) -> str:
    if engine == "LM Studio":
        return _chat_lmstudio(model_name, prompt, temperature, schema)
    kwargs: dict[str, object] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        # Gemma 4 thinks by default and can leave message.content empty,
        # with the only text in message.thinking. This app wants the paper.
        "think": False,
        "options": {"temperature": temperature, "num_predict": 2000},
    }
    if schema is not None:
        kwargs["format"] = schema

    try:
        response = ollama.chat(**kwargs)  # type: ignore[arg-type]
    except Exception as exc:
        raise WriterError(
            f"Ollama call failed for model '{model_name}'. "
            f"Is `ollama serve` running, and is the model pulled? {exc}"
        ) from exc

    message = getattr(response, "message", None)
    content = _reply_text(message)
    if not content:
        reason = getattr(response, "done_reason", None) or "unknown"
        raise WriterError(
            f"Ollama model '{model_name}' returned an empty reply (stop reason: {reason})."
        )
    return content


def _chat_lmstudio(
    model_name: str,
    prompt: str,
    temperature: float,
    schema: dict[str, object] | None,
) -> str:
    payload: dict[str, object] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 2000,
    }
    if schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "outline", "strict": True, "schema": schema},
        }
    try:
        response = _post_json(f"{LMSTUDIO_HOST}/v1/chat/completions", payload)
    except Exception as exc:
        raise WriterError(
            f"LM Studio call failed for model '{model_name}'. "
            f"Is the local server started, and is that model downloaded? {exc}"
        ) from exc

    content, reason = _lmstudio_reply(response)
    if not content:
        raise WriterError(
            f"LM Studio model '{model_name}' returned an empty reply (stop reason: {reason})."
        )
    return content


def _lmstudio_reply(payload: object) -> tuple[str, str]:
    """Visible reply, then a reasoning field if that is all the model returned."""
    if not isinstance(payload, dict):
        return "", "unknown"
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return "", "unknown"
    choice = choices[0]
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    reason = str(choice.get("finish_reason") or "unknown")
    content = message.get("content")
    if content and str(content).strip():
        return str(content).strip(), reason
    for key in ("reasoning_content", "reasoning"):
        extra = message.get(key)
        if extra and str(extra).strip():
            return str(extra).strip(), reason
    return "", reason


def _get_json(url: str, timeout: float = 3) -> object:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _post_json(url: str, payload: dict[str, object], timeout: float = 600) -> object:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _reply_text(message: object) -> str:
    """Prefer the visible reply. Fall back to thinking if that is all Gemma returned."""
    content = getattr(message, "content", None)
    if content and str(content).strip():
        return str(content).strip()
    thinking = getattr(message, "thinking", None)
    if thinking and str(thinking).strip():
        return str(thinking).strip()
    return ""


def _strip_model_heading(text: str) -> str:
    """Drop a leading markdown heading. The screen already prints the section title."""
    lines = text.strip().splitlines()
    if lines and lines[0].lstrip().startswith("#"):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    return "\n".join(lines).strip()


def _parse_outline(raw: str) -> PaperOutline:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if "```" in text:
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    try:
        return PaperOutline.model_validate_json(text)
    except Exception as exc:
        raise WriterError(f"The model did not return a usable outline. {exc}") from exc
