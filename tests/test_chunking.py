import pytest

from app.chunking import chunk_text


def test_short_text_is_one_chunk():
    assert chunk_text("Hello world.", size=100, overlap=10) == ["Hello world."]


def test_paragraphs_are_grouped_up_to_size():
    text = "\n\n".join(["a" * 40] * 5)
    chunks = chunk_text(text, size=100, overlap=0)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(c.replace("\n", "") for c in chunks).count("a") == 200


def test_long_paragraph_is_split_with_overlap():
    text = "".join(str(i % 10) for i in range(1000))
    chunks = chunk_text(text, size=300, overlap=50)
    assert len(chunks) > 3
    assert all(len(c) <= 300 for c in chunks)
    # The end of each chunk reappears at the start of the next one.
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert a[-50:] in b


def test_invalid_arguments():
    with pytest.raises(ValueError):
        chunk_text("x", size=0)
    with pytest.raises(ValueError):
        chunk_text("x", size=10, overlap=10)
