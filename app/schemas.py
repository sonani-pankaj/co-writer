"""Data shapes passed between the screen, the writer, and the scanner.

SectionPlan  - one planned section (heading, shape, points, personal detail)
PaperOutline - title plus 3 or 4 sections
ScanResult   - the numbers shown on the pass/fail bar
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

StructureType = Literal[
    "personal_story_narrative",
    "problem_and_fix",
    "short_bullet_breakdown",
    "real_example_walkthrough",
]


class SectionPlan(BaseModel):
    heading: str
    structure_type: StructureType
    core_points: list[str]
    personal_touch: str = Field(
        description="First-hand experience, test result, or opinion from the user's input"
    )


class PaperOutline(BaseModel):
    title: str
    sections: list[SectionPlan]


class CopyMatch(BaseModel):
    """A sentence that also appears on a public web page."""

    phrase: str
    source: str
    url: str


class ScanResult(BaseModel):
    """Score card for one section.

    ai_probability is a percent (0-100) when the neural detector ran.
    It is null when the detector is not installed; the pass/fail bar then
    uses only banned phrases and burstiness.

    plagiarism_checked stays false until a web phrase search has run.
    That search is not Turnitin. It looks for longer sentences on public pages.
    """

    passed: bool
    ai_probability: float | None
    burstiness: float
    grade_level: float
    flagged_words: list[str]
    detector_used: bool
    plagiarism_checked: bool = False
    plagiarism_error: str = ""
    plagiarism_hits: list[CopyMatch] = Field(default_factory=list)
