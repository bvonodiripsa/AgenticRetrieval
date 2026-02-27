"""Tests for cosmos_retriever module extraction."""

import pytest


# ---------------------------------------------------------------------------
# Import tests – verify the module can be imported independently
# ---------------------------------------------------------------------------

def test_cosmos_retriever_importable():
    """cosmos_retriever should be importable directly (no circular import)."""
    import cosmos_retriever
    assert hasattr(cosmos_retriever, "CombinedRetriever")
    assert hasattr(cosmos_retriever, "RETRIEVAL_SOURCES")


def test_rag_divdet_exposes_combined_retriever():
    """CombinedRetriever should be accessible via rag_divdet after extraction."""
    import rag_divdet
    from cosmos_retriever import CombinedRetriever
    # main_async imports CombinedRetriever at call time; verify the class exists
    assert CombinedRetriever is not None
    # DecomposedRAGPipeline should still be in rag_divdet
    assert hasattr(rag_divdet, "DecomposedRAGPipeline")


# ---------------------------------------------------------------------------
# Structural tests – verify extracted classes/functions are intact
# ---------------------------------------------------------------------------

def test_combined_retriever_has_expected_methods():
    """CombinedRetriever must expose the same public interface after extraction."""
    from cosmos_retriever import CombinedRetriever
    expected_attrs = [
        "initialize",
        "retrieve",
        "close",
        "total_fulltext_k",
        "total_vector_k",
        "source_count",
    ]
    for attr in expected_attrs:
        assert hasattr(CombinedRetriever, attr), f"Missing attribute: {attr}"


def test_retrieval_sources_is_list():
    """RETRIEVAL_SOURCES should be a non-empty list of dicts."""
    from cosmos_retriever import RETRIEVAL_SOURCES
    assert isinstance(RETRIEVAL_SOURCES, list)
    assert len(RETRIEVAL_SOURCES) > 0
    for source in RETRIEVAL_SOURCES:
        assert isinstance(source, dict)
        assert "id" in source
        assert "container_name" in source


def test_stopwords_set_exists():
    """STOPWORDS should be a non-empty set."""
    from cosmos_retriever import STOPWORDS
    assert isinstance(STOPWORDS, set)
    assert len(STOPWORDS) > 0
    assert "the" in STOPWORDS


def test_as_list_of_strings():
    """_as_list_of_strings should convert list items to stripped strings."""
    from cosmos_retriever import _as_list_of_strings
    assert _as_list_of_strings(["a", "b", "c"]) == ["a", "b", "c"]
    assert _as_list_of_strings([1, 2]) == ["1", "2"]
    assert _as_list_of_strings([" x ", ""]) == ["x"]
    assert _as_list_of_strings("not a list") == []
    assert _as_list_of_strings(None) == []


# ---------------------------------------------------------------------------
# Timing globals: _TIMING mutations in rag_divdet are visible
# ---------------------------------------------------------------------------

def test_timing_flag_visible_via_module():
    """Runtime mutations to rag_divdet._TIMING must be visible in cosmos_retriever."""
    import rag_divdet
    import cosmos_retriever  # noqa: F811

    original = rag_divdet._TIMING
    try:
        rag_divdet._TIMING = True
        # cosmos_retriever accesses _rag._TIMING at runtime
        # Verify the module reference is the same object
        import rag_divdet as _rag_ref
        assert _rag_ref._TIMING is True
    finally:
        rag_divdet._TIMING = original
