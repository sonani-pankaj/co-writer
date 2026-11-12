"""Phrase selection and copy matching. No live web search."""

from plagiarism import find_copies, phrases_to_check, words_in_a_row


def test_short_sentences_are_skipped() -> None:
    assert phrases_to_check("Too short. Also tiny.") == []


def test_longest_sentences_come_first() -> None:
    short = "One two three four five six seven eight."
    longer = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda."
    phrases = phrases_to_check(f"{short} {longer}")
    assert phrases[0].startswith("Alpha")


def test_eight_words_in_a_row_count_as_a_copy() -> None:
    phrase = "the team compares small local models like llama and mistral today"
    page = "A blog says the team compares small local models like llama and then stops."
    assert words_in_a_row(phrase, page)


def test_unrelated_page_is_not_a_copy() -> None:
    phrase = "the team compares small local models like llama and mistral today"
    page = "Cooking rice takes water salt and a covered pot on low heat."
    assert words_in_a_row(phrase, page) is False


def test_find_copies_keeps_the_matching_page() -> None:
    phrase = "the team compares small local models like llama and mistral today extra words here"

    def fake_search(query: str) -> list[dict[str, str]]:
        assert query == phrase
        return [
            {
                "title": "A lab note",
                "url": "https://example.com/lab",
                "body": "Notes: the team compares small local models like llama and mistral today.",
            }
        ]

    matches, error = find_copies(phrase + ".", fake_search)
    assert error == ""
    assert matches[0].url == "https://example.com/lab"
    assert "team compares" in matches[0].phrase
