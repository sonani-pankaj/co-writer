"""Local AI-text detector.

The score on screen comes from desklib/ai-text-detector-v1.01 (DeBERTa-v3-large).
It is a local model. It is not Originality, GPTZero, or any other paid detector,
so the percent will not always match those sites. It does catch the modern
model prose that the old ChatGPT-only RoBERTa called "human" at 0%.

The first call downloads about 1.5 GB into the Hugging Face cache.
Set COWRITER_DISABLE_DETECTOR=1 to skip it. Phrase, burstiness, and grade
checks still run.
"""

from __future__ import annotations

import os

from config import DETECTOR_MAX_LENGTH, DETECTOR_MODEL_ID

# Each window is short enough for the model's 768-token limit.
# Overlap keeps a sentence that sits on a cut from being ignored.
_WINDOW_WORDS = 450
_WINDOW_OVERLAP = 80


def load_detector(model_id: str = DETECTOR_MODEL_ID) -> tuple[object | None, str | None]:
    """Return (detect, error). `detect(text)` returns one ChatGPT-label score, 0 to 1."""
    if os.environ.get("COWRITER_DISABLE_DETECTOR") == "1":
        return None, "Detector is off (COWRITER_DISABLE_DETECTOR=1)."

    try:
        import torch
        import torch.nn as nn
        from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel
    except Exception as exc:
        return None, f"Detector dependencies are not importable: {exc}"

    class DesklibAIDetectionModel(PreTrainedModel):
        config_class = AutoConfig

        def __init__(self, config):
            super().__init__(config)
            self.model = AutoModel.from_config(config)
            self.classifier = nn.Linear(config.hidden_size, 1)
            self.post_init()

        def forward(self, input_ids, attention_mask=None, labels=None):
            outputs = self.model(input_ids, attention_mask=attention_mask)
            hidden = outputs[0]
            mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            pooled = torch.sum(hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            return {"logits": self.classifier(pooled)}

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = DesklibAIDetectionModel.from_pretrained(model_id)
        model.eval()
    except Exception as exc:
        return None, f"Could not load {model_id}: {exc}"

    def detect(text: str) -> list[dict[str, object]]:
        probability = _max_ai_probability(text, model, tokenizer, torch)
        # Scanner treats the ChatGPT label score as the AI probability.
        return [{"label": "ChatGPT", "score": probability}]

    return detect, None


def text_windows(text: str) -> list[str]:
    """Split long text so each piece fits the detector. Short text stays one piece."""
    words = text.split()
    if len(words) <= _WINDOW_WORDS:
        return [text.strip()] if text.strip() else []

    step = _WINDOW_WORDS - _WINDOW_OVERLAP
    windows: list[str] = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + _WINDOW_WORDS]).strip()
        if chunk:
            windows.append(chunk)
        if start + _WINDOW_WORDS >= len(words):
            break
    return windows


def _max_ai_probability(text: str, model, tokenizer, torch) -> float:
    """Highest AI probability across windows. One confident window is enough to flag."""
    windows = text_windows(text)
    if not windows:
        return 0.0

    highest = 0.0
    for window in windows:
        encoded = tokenizer(
            window,
            padding="max_length",
            truncation=True,
            max_length=DETECTOR_MAX_LENGTH,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )["logits"]
            highest = max(highest, float(torch.sigmoid(logits).item()))
    return highest
