"""The screen. Run this file; it does not contain scoring or prompt text.

    uv run streamlit run app/main.py

Sidebar: writer (Ollama or LM Studio), downloaded models, audience, English level, then papers you already generated.
Main column, full width: notes, then the paper underneath.
"""

from __future__ import annotations

import streamlit as st

from config import AUDIENCES, DETECTOR_MODEL_ID, ENGLISH_LEVELS, MIN_BURSTINESS
from detector import load_detector
from plagiarism import find_copies, search_web
from scanner import scan_text
from schemas import ScanResult, SectionPlan
from writer import (
    ENGINES,
    WriterError,
    build_outline,
    draft_section,
    rewrite_section,
    write_paragraph,
    writer_status,
)


st.set_page_config(page_title="Co-writer", layout="wide")


@st.cache_resource(show_spinner="Loading the local AI detector (first run downloads about 1.5 GB)...")
def cached_detector(model_id: str) -> tuple[object | None, str | None]:
    return load_detector(model_id)


def _rescore_open_sections(detector: object | None) -> None:
    """Recompute scores when the detector changes, so an old 0% does not stay on screen."""
    marker = DETECTOR_MODEL_ID if detector is not None else "off"
    if st.session_state.get("scored_with") == marker:
        return
    sections = st.session_state.get("sections_data") or []
    if sections:
        with st.spinner("Scoring the open sections again..."):
            for index, item in enumerate(sections):
                text = st.session_state.get(f"sec_text_{index}", item["text"])
                item["text"] = text
                item["stats"] = scan_text(text, detector)
    st.session_state["scored_with"] = marker


def _init_state() -> None:
    if "paper_title" not in st.session_state:
        st.session_state.paper_title = ""
    if "sections_data" not in st.session_state:
        st.session_state.sections_data = []
    if "history" not in st.session_state:
        st.session_state.history = []
    if "active_history" not in st.session_state:
        st.session_state.active_history = None
    if "paragraph_stats" not in st.session_state:
        st.session_state.paragraph_stats = None


def _clear_section_widgets() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith(("sec_text_", "sec_hint_", "rewrite_requested_", "rewrite_error_")):
            del st.session_state[key]


def _generate_paper(
    rough_notes: str,
    experience_notes: str,
    audience: str,
    model_name: str,
    english_level: str,
    engine: str,
) -> None:
    with st.spinner("Planning varied sections from your notes..."):
        outline = build_outline(
            rough_notes, experience_notes, audience, model_name, english_level, engine=engine
        )

    _clear_section_widgets()
    st.session_state.paper_title = outline.title
    st.session_state.sections_data = []

    detector, _error = cached_detector(DETECTOR_MODEL_ID)
    for index, section in enumerate(outline.sections):
        with st.spinner(f"Drafting section {index + 1}: {section.heading}..."):
            draft = draft_section(section, audience, model_name, english_level, engine=engine)
        stats = scan_text(draft, detector)
        st.session_state.sections_data.append({"plan": section, "text": draft, "stats": stats})
        st.session_state[f"sec_text_{index}"] = draft

    _remember_paper(model_name, audience, engine)


def _apply_pending_rewrite(
    index: int,
    audience: str,
    model_name: str,
    english_level: str,
    engine: str,
    detector: object | None,
) -> None:
    """Rewrite must change the text box before that box is drawn on this run."""
    if not st.session_state.pop(f"rewrite_requested_{index}", False):
        return

    item = st.session_state.sections_data[index]
    current_text = st.session_state.get(f"sec_text_{index}", item["text"])
    extra_hint = st.session_state.get(f"sec_hint_{index}", "")
    try:
        with st.spinner(f"Rewriting section {index + 1}..."):
            new_text = rewrite_section(
                current_text=current_text,
                section=item["plan"],
                stats=item["stats"],
                audience=audience,
                model_name=model_name,
                english_level=english_level,
                extra_hint=extra_hint,
                engine=engine,
            )
    except WriterError as exc:
        st.session_state[f"rewrite_error_{index}"] = str(exc)
        return

    stats = scan_text(new_text, detector)
    st.session_state[f"sec_text_{index}"] = new_text
    st.session_state.sections_data[index]["text"] = new_text
    st.session_state.sections_data[index]["stats"] = stats
    st.session_state.pop(f"rewrite_error_{index}", None)


def _on_manual_edit(index: int) -> None:
    detector, _error = cached_detector(DETECTOR_MODEL_ID)
    updated = st.session_state[f"sec_text_{index}"]
    st.session_state.sections_data[index]["text"] = updated
    st.session_state.sections_data[index]["stats"] = scan_text(updated, detector)


def _plagiarism_label(stats: ScanResult) -> str:
    if getattr(stats, "plagiarism_error", ""):
        return "search failed"
    if not getattr(stats, "plagiarism_checked", False):
        return "not checked"
    hits = getattr(stats, "plagiarism_hits", [])
    if not hits:
        return "no web matches"
    return f"{len(hits)} possible copies"


def _show_plagiarism_hits(stats: ScanResult) -> None:
    for hit in getattr(stats, "plagiarism_hits", []):
        st.markdown(f"- “{hit.phrase}” — [{hit.source}]({hit.url})")
    error = getattr(stats, "plagiarism_error", "")
    if error:
        st.caption(f"Plagiarism search stopped: {error}")


def _attach_plagiarism(stats: ScanResult, text: str) -> ScanResult:
    matches, error = find_copies(text, search_web)
    stats.plagiarism_checked = True
    stats.plagiarism_error = error
    stats.plagiarism_hits = matches
    return stats


def _score_banner(stats: ScanResult, structure_type: str) -> None:
    ai_bit = "n/a (detector off)" if stats.ai_probability is None else f"{stats.ai_probability}% local"
    copy_bit = _plagiarism_label(stats)
    line = (
        f"AI score: {ai_bit} | "
        f"Burstiness: {stats.burstiness} | "
        f"Grade level: {stats.grade_level} | "
        f"Plagiarism: {copy_bit} | Style: {structure_type}"
    )
    if stats.passed:
        st.success(f"PASS | {line}")
    else:
        st.warning(
            f"HIGH AI / UNIFORM FLAG | {line} | "
            f"Burstiness target >= {MIN_BURSTINESS} | "
            f"Flagged words: {stats.flagged_words or 'none'}"
        )
    _show_plagiarism_hits(stats)


def _snapshot_sections() -> list[dict[str, object]]:
    snapshot = []
    for index, item in enumerate(st.session_state.sections_data):
        snapshot.append(
            {
                "plan": item["plan"],
                "text": st.session_state.get(f"sec_text_{index}", item["text"]),
                "stats": item["stats"],
            }
        )
    return snapshot


def _remember_paper(model_name: str, audience: str, engine: str) -> None:
    """Keep this paper in the sidebar. Newest paper is first."""
    st.session_state.history.insert(
        0,
        {
            "title": st.session_state.paper_title,
            "model": model_name,
            "engine": engine,
            "audience": audience,
            "sections": _snapshot_sections(),
        },
    )
    st.session_state.history = st.session_state.history[:20]
    st.session_state.active_history = 0


def _persist_active() -> None:
    index = st.session_state.get("active_history")
    history = st.session_state.get("history") or []
    if index is None or not history or not (0 <= index < len(history)):
        return
    if not st.session_state.sections_data:
        return
    history[index]["title"] = st.session_state.paper_title
    history[index]["sections"] = _snapshot_sections()


def _open_history(index: int) -> None:
    _persist_active()
    entry = st.session_state.history[index]
    _clear_section_widgets()
    st.session_state.paper_title = entry["title"]
    st.session_state.sections_data = [
        {"plan": section["plan"], "text": section["text"], "stats": section["stats"]}
        for section in entry["sections"]
    ]
    for section_index, section in enumerate(entry["sections"]):
        st.session_state[f"sec_text_{section_index}"] = section["text"]
    st.session_state.active_history = index


def _draw_sidebar(
    detector: object | None,
    detector_error: str | None,
) -> tuple[str, str, str, str]:
    engine = st.sidebar.radio("Writer", list(ENGINES), horizontal=True, key="writer_engine")
    is_up, installed, detail = writer_status(engine)
    _writer_status_icons(engine, is_up, installed, detail, detector, detector_error)

    model_name = _selected_model(installed if is_up else [], engine)
    audience = st.sidebar.selectbox("Target audience", AUDIENCES, key="audience")
    level_name = st.sidebar.selectbox("English level", list(ENGLISH_LEVELS), key="english_level")
    st.sidebar.caption(ENGLISH_LEVELS[level_name])

    st.sidebar.header("History")
    history = st.session_state.history
    if not history:
        st.sidebar.caption("Papers you generate are listed here.")
        return engine, model_name, audience, ENGLISH_LEVELS[level_name]
    for index, entry in enumerate(history):
        label = entry["title"] or "Untitled paper"
        st.sidebar.button(
            label,
            key=f"history_btn_{index}",
            use_container_width=True,
            type="primary" if index == st.session_state.active_history else "secondary",
            on_click=_open_history,
            args=(index,),
        )
        st.sidebar.caption(f"{entry.get('engine', 'Ollama')} · {entry['model']}")
    return engine, model_name, audience, ENGLISH_LEVELS[level_name]


def _compiled_markdown() -> str:
    parts = [f"# {st.session_state.paper_title}", ""]
    for index, item in enumerate(st.session_state.sections_data):
        heading = item["plan"].heading
        body = st.session_state.get(f"sec_text_{index}", item["text"])
        parts.append(f"## {heading}\n{body}")
    return "\n\n".join(parts).strip() + "\n"


def _writer_status_icons(
    engine: str,
    is_up: bool,
    installed: list[str],
    detail: str,
    detector: object | None,
    detector_error: str | None,
) -> None:
    """One icon row: server, downloaded models, detector. Hover for the words."""
    count = len(installed)
    notes: list[str] = []
    if is_up:
        server = ":green[:material/check_circle:]"
        notes.append(f"{engine} is running.")
    else:
        server = ":red[:material/cloud_off:]"
        notes.append(f"{engine} is not reachable.")
        if detail.strip():
            notes.append(detail.strip())
    notes.append(f"{count} downloaded model{'s' if count != 1 else ''}.")
    if detector is None:
        shield = ":orange[:material/shield:]"
        notes.append(detector_error or "Local detector is off.")
    else:
        shield = ":green[:material/shield:]"
        notes.append(f"Detector: {DETECTOR_MODEL_ID}")
    st.sidebar.markdown(
        f"{server} &nbsp; :material/download: {count} &nbsp; {shield}",
        help=" ".join(notes),
    )


def _selected_model(installed: list[str], engine: str) -> str:
    """Dropdown options are the models already downloaded for the chosen writer."""
    if not installed:
        if engine == "LM Studio":
            st.sidebar.caption("No downloaded models yet. Download one in LM Studio, then reload this page.")
        else:
            st.sidebar.caption("No downloaded models yet. Run `ollama pull <name>`, then reload this page.")
        return ""

    # The previous choice may belong to the other writer, or may have been removed.
    # Clear it before the selectbox is drawn, or Streamlit rejects the value.
    if st.session_state.get("model_choice") not in installed:
        st.session_state["model_choice"] = installed[0]
    return st.sidebar.selectbox("Downloaded model", installed, key="model_choice")


def _apply_section_rescan(index: int, detector: object | None) -> None:
    """Re-scan runs before the score bar, including the web plagiarism check."""
    if not st.session_state.pop(f"plagiarism_rescan_{index}", False):
        return
    text = st.session_state.get(f"sec_text_{index}", st.session_state.sections_data[index]["text"])
    stats = scan_text(text, detector)
    with st.spinner("Checking the web for copied sentences..."):
        _attach_plagiarism(stats, text)
    st.session_state.sections_data[index]["text"] = text
    st.session_state.sections_data[index]["stats"] = stats


def _request_section_rescan(index: int) -> None:
    st.session_state[f"plagiarism_rescan_{index}"] = True


def _draw_section(
    index: int,
    audience: str,
    model_name: str,
    english_level: str,
    engine: str,
    detector: object | None,
) -> None:
    _apply_pending_rewrite(index, audience, model_name, english_level, engine, detector)
    _apply_section_rescan(index, detector)

    item = st.session_state.sections_data[index]
    plan: SectionPlan = item["plan"]
    stats: ScanResult = item["stats"]

    with st.container(border=True):
        st.subheader(f"{index + 1}. {plan.heading}")
        _score_banner(stats, plan.structure_type)

        error = st.session_state.get(f"rewrite_error_{index}")
        if error:
            st.error(error)

        st.text_area(
            "Edit this section (click outside the box to score it again):",
            key=f"sec_text_{index}",
            height=180,
            on_change=_on_manual_edit,
            args=(index,),
        )

        hint_col, rewrite_col, rescan_col = st.columns([2, 1, 1], vertical_alignment="bottom")
        with hint_col:
            st.text_input(
                "Personal detail to add on the next rewrite (rough English is fine):",
                key=f"sec_hint_{index}",
                placeholder="e.g. mention we lost 2 days fixing a Redis cache bug",
            )
        with rewrite_col:
            st.button(
                "Rewrite section",
                key=f"rewrite_btn_{index}",
                type="primary" if not stats.passed else "secondary",
                on_click=_request_rewrite,
                args=(index,),
            )
        with rescan_col:
            st.button(
                "Re-scan score",
                key=f"rescan_btn_{index}",
                on_click=_request_section_rescan,
                args=(index,),
            )


def _write_checked_paragraph(
    rough_notes: str,
    audience: str,
    model_name: str,
    english_level: str,
    line_count: int,
    engine: str,
) -> None:
    detector, _error = cached_detector(DETECTOR_MODEL_ID)
    with st.spinner("Writing one paragraph..."):
        text = write_paragraph(
            rough_notes, audience, model_name, english_level, line_count, engine=engine
        )
    text = " ".join(text.split())
    stats = scan_text(text, detector)
    with st.spinner("Checking the web for copied sentences..."):
        _attach_plagiarism(stats, text)
    st.session_state.paragraph_text = text
    st.session_state.paragraph_lines = line_count
    st.session_state.paragraph_stats = stats


def _request_paragraph_rescan() -> None:
    st.session_state.paragraph_plagiarism = True


def _apply_paragraph_rescan(detector: object | None) -> None:
    if not st.session_state.pop("paragraph_plagiarism", False):
        return
    text = " ".join(st.session_state.get("paragraph_text", "").split())
    st.session_state.paragraph_text = text
    stats = scan_text(text, detector)
    with st.spinner("Checking the web for copied sentences..."):
        _attach_plagiarism(stats, text)
    st.session_state.paragraph_stats = stats


def _on_paragraph_edit() -> None:
    detector, _error = cached_detector(DETECTOR_MODEL_ID)
    st.session_state.paragraph_stats = scan_text(st.session_state.paragraph_text, detector)


def _request_rewrite(index: int) -> None:
    st.session_state[f"rewrite_requested_{index}"] = True
    st.session_state.pop(f"rewrite_error_{index}", None)


def main() -> None:
    _init_state()
    detector, detector_error = cached_detector(DETECTOR_MODEL_ID)
    _rescore_open_sections(detector)

    st.title("Co-writer")
    st.caption(
        "Turn rough notes into a paper or one paragraph. "
        "The score bar checks phrases, sentence variety, reading grade, and the local detector."
    )

    engine, model_name, audience, english_level = _draw_sidebar(detector, detector_error)

    write_mode = st.radio("What to write", ["Paper", "Paragraph"], horizontal=True, key="write_mode")
    rough_input = st.text_area(
        "Idea, in rough English or bullet points",
        height=140,
        placeholder=(
            "Example: I want write about why microservice fail when team small. "
            "Too much docker container, network slow, hard debug..."
        ),
    )
    can_write = bool(rough_input.strip()) and bool(model_name)

    if write_mode == "Paper":
        personal_input = st.text_area(
            "One or two real experiences or numbers",
            height=100,
            placeholder=(
                "Example: Last year my 4 person team split one app into 12 services. "
                "Deployment took 3 hours instead of 10 minutes..."
            ),
        )
        generate = st.button("Generate paper", type="primary", disabled=not can_write)
        if generate:
            try:
                _generate_paper(
                    rough_input, personal_input, audience, model_name, english_level, engine
                )
            except WriterError as exc:
                st.error(str(exc))

        st.divider()
        if not st.session_state.sections_data:
            st.info("The paper shows up here after you generate it. You can edit every section.")
        else:
            st.header(st.session_state.paper_title)
            for index in range(len(st.session_state.sections_data)):
                _draw_section(index, audience, model_name, english_level, engine, detector)
            st.download_button(
                "Download paper (.md)",
                data=_compiled_markdown(),
                file_name="paper.md",
                mime="text/markdown",
                type="primary",
            )
    else:
        line_count = int(
            st.number_input(
                "Number of lines",
                min_value=1,
                max_value=30,
                value=5,
                step=1,
                help="Each line is about 12 words. The result is one paragraph of that length.",
            )
        )
        write_col, rescan_col, _spacer = st.columns([1, 1, 2], vertical_alignment="bottom")
        with write_col:
            write_one = st.button(
                "Write paragraph",
                type="primary",
                disabled=not can_write,
                use_container_width=True,
            )
        with rescan_col:
            rescan_paragraph = st.button(
                "Re-scan score",
                disabled=st.session_state.paragraph_stats is None,
                use_container_width=True,
            )
        if write_one:
            try:
                _write_checked_paragraph(
                    rough_input, audience, model_name, english_level, line_count, engine
                )
            except WriterError as exc:
                st.error(str(exc))
        if rescan_paragraph:
            _request_paragraph_rescan()
        _apply_paragraph_rescan(detector)

        st.divider()
        paragraph_stats = st.session_state.paragraph_stats
        if paragraph_stats is None:
            st.info("The paragraph shows up here. The score bar checks it after it is written.")
        else:
            asked = int(st.session_state.get("paragraph_lines") or line_count)
            current = st.session_state.get("paragraph_text", "")
            if "\n" in current:
                st.session_state.paragraph_text = " ".join(current.split())
            body = st.session_state.get("paragraph_text", "")
            st.subheader("Paragraph")
            st.caption(f"{len(body.split())} words. You asked for {asked} lines, about {asked * 12} words.")
            _score_banner(paragraph_stats, "paragraph")
            st.text_area(
                "Edit the paragraph (click outside the box to score it again):",
                key="paragraph_text",
                height=220,
                on_change=_on_paragraph_edit,
            )


main()
