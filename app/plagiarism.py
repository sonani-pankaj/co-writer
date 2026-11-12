"""Find sentences in the user's text that also show up on public web pages.

This is a phrase search, not a Turnitin database. It checks the longest
sentences against DuckDuckGo and keeps a hit when the same run of words
appears in a result.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from schemas import CopyMatch

_SENTENCE_SPLIT = re.compile(r"[.!?]+")
_WORD = re.compile(r"[a-z0-9']+")

# Ignore tiny sentences. A copied line is usually at least this long.
_MIN_WORDS = 8
_MAX_PHRASES = 4
# This many words in a row, found in a search result, counts as a copy.
# Six is enough for a distinctive clause and still shows up in short snippets.
_RUN = 6


def phrases_to_check(text: str) -> list[str]:
    """Longest sentences, up to four. Short fragments are skipped."""
    sentences = []
    for part in _SENTENCE_SPLIT.split(text):
        sentence = " ".join(part.split())
        if len(_words(sentence)) >= _MIN_WORDS:
            sentences.append(sentence)
    sentences.sort(key=lambda sentence: len(_words(sentence)), reverse=True)
    return sentences[:_MAX_PHRASES]


def words_in_a_row(phrase: str, haystack: str, run: int = _RUN) -> bool:
    """True when `run` words from the phrase appear in order in the haystack."""
    phrase_words = _words(phrase)
    page_words = _words(haystack)
    if len(phrase_words) < run or len(page_words) < run:
        return False
    for start in range(0, len(phrase_words) - run + 1):
        window = phrase_words[start : start + run]
        for offset in range(0, len(page_words) - run + 1):
            if page_words[offset : offset + run] == window:
                return True
    return False


def find_copies(text: str, search: Callable[[str], list[dict[str, str]]]) -> tuple[list[CopyMatch], str]:
    """Return (matches, error). `error` is set when the search itself fails."""
    phrases = phrases_to_check(text)
    if not phrases:
        return [], ""

    matches: list[CopyMatch] = []
    seen_urls: set[str] = set()
    try:
        for phrase in phrases:
            for result in search(phrase):
                blob = f"{result.get('title', '')} {result.get('body', '')}"
                url = result.get("url", "")
                if not url or url in seen_urls:
                    continue
                if words_in_a_row(phrase, blob):
                    seen_urls.add(url)
                    matches.append(
                        CopyMatch(
                            phrase=phrase,
                            source=result.get("title") or url,
                            url=url,
                        )
                    )
                    break
    except Exception as exc:
        return matches, str(exc)
    return matches, ""


def search_web(phrase: str) -> list[dict[str, str]]:
    """Quoted search. Returns title, url, and snippet for each result."""
    from ddgs import DDGS

    query = f'"{phrase}"'
    try:
        items = list(DDGS().text(query, max_results=5))
    except Exception as exc:
        if "No results" in str(exc):
            return []
        raise
    return [
        {
            "title": str(item.get("title") or ""),
            "url": str(item.get("href") or item.get("url") or ""),
            "body": str(item.get("body") or ""),
        }
        for item in items
    ]


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())
