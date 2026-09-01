"""Tests for the RAG pipeline: documents, query analysis, retrieval, chatbot.

The retrieval and analysis layers are tested thoroughly because they are fast
and deterministic.  The handful of tests that actually run the language model
are marked ``llm`` so they can be deselected with ``-m "not llm"`` when a quick
run is wanted.
"""

from __future__ import annotations

import pytest

from src.rag.documents import build_documents_for_phone, build_overview_text
from src.rag.query_analyzer import QueryIntent, analyze_query
from src.rag.vector_store import HybridVectorStore, tokenize


# ---------------------------------------------------------------------------
# Query analysis (no database, no model)
# ---------------------------------------------------------------------------
class TestQueryAnalysisWithoutDatabase:
    @pytest.mark.parametrize(
        "question,expected",
        [
            ("Which Samsung phone has the best battery life?", QueryIntent.SUPERLATIVE),
            ("Which phone is the lightest?", QueryIntent.SUPERLATIVE),
            ("Which has the most RAM?", QueryIntent.SUPERLATIVE),
            ("Recommend a Samsung phone for gaming", QueryIntent.RECOMMENDATION),
            ("Hello there", QueryIntent.GENERAL),
        ],
    )
    def test_intents_that_need_no_phone_lookup(self, question, expected):
        assert analyze_query(question).intent is expected

    @pytest.mark.parametrize(
        "question,column",
        [
            ("Which phone has the best battery life?", "battery_capacity_mah"),
            ("biggest screen", "display_size_inches"),
            ("most megapixels", "main_camera_mp"),
            ("most storage", "max_storage_gb"),
            ("fastest charging", "charging_watts"),
            ("lightest phone", "weight_grams"),
        ],
    )
    def test_ranking_column_detection(self, question, column):
        assert analyze_query(question).ranking_column == column

    def test_direction_for_minimising_superlatives(self):
        assert analyze_query("Which phone is the lightest?").descending is False
        assert analyze_query("Which phone has the biggest battery?").descending is True


class TestQueryAnalysisWithDatabase:
    def test_single_phone_is_resolved(self, populated_repository):
        analysis = analyze_query(
            "What are the camera specs of the Samsung Galaxy S23?",
            populated_repository,
        )
        assert analysis.intent is QueryIntent.SPEC_LOOKUP
        assert len(analysis.matched_phones) == 1
        assert "S23" in analysis.matched_phones[0].name

    def test_comparison_resolves_both_phones(self, populated_repository):
        analysis = analyze_query(
            "How does the Galaxy S23 compare to the S22 in terms of performance?",
            populated_repository,
        )
        assert analysis.intent is QueryIntent.COMPARISON
        assert len(analysis.matched_phones) == 2

    def test_shorthand_vs_syntax(self, populated_repository):
        analysis = analyze_query("S24 Ultra vs S23 Ultra", populated_repository)
        assert analysis.intent is QueryIntent.COMPARISON
        names = {phone.name for phone in analysis.matched_phones}
        assert len(names) == 2

    def test_foldable_naming(self, populated_repository):
        analysis = analyze_query("Tell me about the Z Fold5", populated_repository)
        assert analysis.matched_phones
        assert "Fold5" in analysis.matched_phones[0].name

    def test_unknown_model_resolves_to_nothing(self, populated_repository):
        analysis = analyze_query("What about the iPhone 15?", populated_repository)
        assert analysis.matched_phones == []


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
class TestDocumentBuilding:
    def test_overview_mentions_the_phone_and_key_numbers(self, populated_repository):
        phone = populated_repository.find_by_name("Galaxy S23")
        text = build_overview_text(phone)
        assert phone.name in text
        assert str(phone.battery_capacity_mah) in text
        assert str(phone.display_size_inches) in text

    def test_each_document_names_its_phone(self, populated_repository):
        phone = populated_repository.find_by_name("Galaxy S23")
        grouped = populated_repository.specs_by_category(phone)
        documents = build_documents_for_phone(phone, grouped)

        assert len(documents) > 5
        for document in documents:
            assert phone.name in document.text
            assert document.phone_id == phone.id
            assert document.text.strip()

    def test_camera_lenses_are_numbered_separately(self, populated_repository):
        """Each lens must stay distinct - see GSMArenaScraper._cell_text."""
        phone = populated_repository.find_by_name("Galaxy S23 Ultra")
        grouped = populated_repository.specs_by_category(phone)
        documents = build_documents_for_phone(phone, grouped)

        camera = next(d for d in documents if d.section == "Main Camera")
        assert "Lens 1:" in camera.text
        assert "Lens 2:" in camera.text

    def test_document_ids_are_unique(self, populated_repository):
        from src.rag.documents import build_corpus

        documents = build_corpus(populated_repository)
        ids = [document.doc_id for document in documents]
        assert len(set(ids)) == len(ids)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
class TestTokenizer:
    def test_keeps_model_numbers_and_units_intact(self):
        assert tokenize("Galaxy S23 Ultra 120Hz 5000mAh") == [
            "galaxy",
            "s23",
            "ultra",
            "120hz",
            "5000mah",
        ]


@pytest.fixture(scope="module")
def vector_store():
    store = HybridVectorStore()
    if not store.load():
        pytest.skip("No vector store - run 'python -m scripts.build_index' first")
    return store


class TestHybridRetrieval:
    def test_store_contains_documents(self, vector_store):
        assert len(vector_store) > 50

    def test_search_returns_scored_results(self, vector_store):
        results = vector_store.search("battery capacity", top_k=5)
        assert len(results) == 5
        scores = [score for _document, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_named_model_retrieves_that_model(self, vector_store):
        results = vector_store.search("Galaxy S23 Ultra camera", top_k=5)
        names = {document.phone_name for document, _score in results}
        assert any("S23 Ultra" in name for name in names)

    def test_exact_chipset_token_is_found(self, vector_store):
        """BM25 half of the hybrid: exact technical tokens must match."""
        results = vector_store.search("Snapdragon 8 Gen 2", top_k=5)
        assert any(
            "Snapdragon 8 Gen 2" in document.text for document, _score in results
        )

    def test_weak_paraphrase_is_covered_by_the_structured_path(
        self, populated_repository
    ):
        """Vector search alone is unreliable on loose paraphrases.

        "which phone lasts longest on a charge" shares no vocabulary with the
        indexed text and retrieves poorly.  The query analyser still routes it
        to a SQL ranking, which is the whole reason the system does not rely on
        embeddings alone for superlatives.
        """
        analysis = analyze_query(
            "which phone lasts longest on a charge", populated_repository
        )
        assert analysis.intent is QueryIntent.SUPERLATIVE
        assert analysis.ranking_column == "battery_capacity_mah"

        ranked = populated_repository.rank_by_battery_life(limit=3)
        assert ranked
        assert "S25 Ultra" in ranked[0].name

    def test_phone_filter_restricts_results(self, vector_store):
        target = vector_store.documents[0].phone_id
        results = vector_store.search("display", top_k=4, phone_ids={target})
        assert all(document.phone_id == target for document, _score in results)

    def test_semantic_question_finds_battery_documents(self, vector_store):
        """A paraphrase with no shared keywords must still reach battery data.

        "Battery" appears in the Battery section and in the Our Tests section
        (which carries the measured endurance score), so either is a hit.
        """
        results = vector_store.search("how long does the phone last", top_k=6)
        assert any(
            "battery" in document.text.lower() for document, _score in results
        )


# ---------------------------------------------------------------------------
# End-to-end chatbot (runs the language model)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def chatbot(request):
    from src.rag.chatbot import SamsungChatbot

    store = HybridVectorStore()
    if not store.load():
        pytest.skip("No vector store - run 'python -m scripts.build_index' first")
    return SamsungChatbot(vector_store=store)


@pytest.mark.llm
class TestChatbot:
    def test_spec_question_returns_the_right_figure(self, chatbot):
        response = chatbot.ask("What is the screen size of the Galaxy S22?")
        assert "6.1" in response.answer
        assert response.intent == QueryIntent.SPEC_LOOKUP.value
        assert response.sources

    def test_superlative_question_names_the_top_phone(self, chatbot):
        response = chatbot.ask("Which Samsung phone has the best battery life?")
        assert response.intent == QueryIntent.SUPERLATIVE.value
        assert "S25 Ultra" in response.answer

    def test_comparison_reports_both_phones(self, chatbot):
        response = chatbot.ask(
            "How does the Galaxy S23 compare to the S22 in terms of performance?"
        )
        assert response.intent == QueryIntent.COMPARISON.value
        assert len(response.phones_referenced) == 2

    def test_empty_question_is_handled(self, chatbot):
        response = chatbot.ask("   ")
        assert response.answer
        assert not response.sources
