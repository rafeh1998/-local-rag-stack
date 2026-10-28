import pytest

from app.store import VectorStore


def test_search_returns_closest_first(tmp_path):
    s = VectorStore(tmp_path)
    s.add("doc", ["north", "east"], [[1, 0], [0, 1]])
    hits = s.search([0.9, 0.1], k=2)
    assert [h.text for h in hits] == ["north", "east"]
    assert hits[0].score > hits[1].score


def test_index_survives_restart(tmp_path):
    VectorStore(tmp_path).add("doc", ["hello"], [[1, 2, 3]])
    reloaded = VectorStore(tmp_path)
    assert len(reloaded) == 1
    assert reloaded.search([1, 2, 3], k=1)[0].text == "hello"


def test_delete_source(tmp_path):
    s = VectorStore(tmp_path)
    s.add("a", ["x"], [[1, 0]])
    s.add("b", ["y"], [[0, 1]])
    assert s.delete_source("a") == 1
    assert s.sources == ["b"]
    assert s.delete_source("b") == 1
    assert len(s) == 0
    assert s.search([1, 0]) == []


def test_dimension_mismatch_is_rejected(tmp_path):
    s = VectorStore(tmp_path)
    s.add("a", ["x"], [[1, 0]])
    with pytest.raises(ValueError):
        s.add("b", ["y"], [[1, 0, 0]])
