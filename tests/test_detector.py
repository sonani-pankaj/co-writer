"""Window splitting. Does not download the neural model."""

from detector import text_windows


def test_short_text_is_one_window() -> None:
    assert text_windows("one two three") == ["one two three"]


def test_blank_text_has_no_windows() -> None:
    assert text_windows("   ") == []


def test_long_text_overlaps() -> None:
    words = [f"w{i}" for i in range(500)]
    windows = text_windows(" ".join(words))
    assert len(windows) >= 2
    assert windows[0].split()[0] == "w0"
    assert windows[-1].split()[-1] == "w499"
