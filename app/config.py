"""Settings you can change without reading the rest of the app.

Audiences, banned phrases, and the pass/fail limits live here.
The model dropdown is not here. It is filled from `ollama list` at runtime.
"""

from __future__ import annotations

AUDIENCES: list[str] = [
    "General Readers (Simple & Clear)",
    "Engineers / Developers",
    "Business Managers",
    "Academic / Students",
]

# CEFR band -> rule pasted into the outline, draft, and rewrite prompts.
# The sidebar shows the key. The value is what the model must follow.
ENGLISH_LEVELS: dict[str, str] = {
    "A1–A2 (Basic)": (
        "Write at CEFR A1–A2, a basic user. Use short, simple sentences and "
        "separate phrases. Everyday words only. This should read like a basic "
        "note or a few personal details. Do not use long sentences, rare words, or idioms."
    ),
    "B1–B2 (Independent)": (
        "Write at CEFR B1–B2, an independent user. Write connected text on a "
        "familiar topic. A B2 text is clear and detailed, and it can state a "
        "viewpoint. Prefer common words. Do not turn it into academic prose."
    ),
    "C1–C2 (Proficient)": (
        "Write at CEFR C1–C2, a proficient user. C1 is clear and well structured "
        "on a complex subject. C2 is precise and natural, and the style fits the "
        "reader. Stay concrete. Do not become stiff, vague, or overly formal."
    ),
}

# Substring match, lowercased. If any of these appear, the section fails.
BANNED_AI_PHRASES: list[str] = [
    "in conclusion",
    "in today's world",
    "in the digital landscape",
    "let's delve into",
    "delve",
    "moreover",
    "furthermore",
    "in order to",
    "by doing this",
    "it is important to note",
    "testament to",
    "tapestry",
    "actions speak louder than words",
    "firstly",
    "secondly",
    "thirdly",
]

# Sentence-length variation. Sample standard deviation of words per sentence.
# Below this, the section looks too uniform and fails the scan.
MIN_BURSTINESS = 4.2

# Hugging Face label confidence treated as "this was written by ChatGPT".
# Stored internally as 0.0-1.0. The screen shows it as a percent.
MAX_AI_PROBABILITY = 0.65

# Numbers written into the rewrite prompt. They are stricter than the pass line
# so a rewrite aims past the minimum instead of landing on it.
REWRITE_TARGET_AI_PERCENT = 50
REWRITE_TARGET_BURSTINESS = 4.5

# Local detector. Not Originality. Skipped when COWRITER_DISABLE_DETECTOR=1.
DETECTOR_MODEL_ID = "desklib/ai-text-detector-v1.01"
DETECTOR_MAX_LENGTH = 768

# Labels that mean "AI-written". Any other label is treated as human,
# and the AI probability becomes (1 - confidence).
AI_LABELS = frozenset({"ChatGPT", "Fake", "LABEL_1", "AI"})
