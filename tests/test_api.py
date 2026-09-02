"""Tests for the FastAPI layer.

The tests run against a real application instance through FastAPI's
``TestClient``, which triggers the normal start-up sequence.  Endpoints that
invoke the language model are marked ``llm`` so the fast checks can be run on
their own with ``-m "not llm"``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client(request):
    if not request.getfixturevalue("database_available"):
        pytest.skip("MySQL server is not reachable")

    from src.database.connection import session_scope
    from src.database.repository import PhoneRepository

    with session_scope() as session:
        if PhoneRepository(session).count_phones() == 0:
            pytest.skip("Database is empty - run the scraper first")

    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Service endpoints
# ---------------------------------------------------------------------------
class TestServiceEndpoints:
    def test_api_index_lists_the_endpoints(self, client):
        response = client.get("/api")
        assert response.status_code == 200
        body = response.json()
        assert "endpoints" in body
        assert body["service"]

    def test_root_serves_the_web_client(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "Samsung Phone Query and Review System" in response.text

    def test_web_client_only_calls_documented_endpoints(self, client):
        """The page must not depend on routes the API does not expose."""
        page = client.get("/").text
        served = {
            route.path for route in app.routes if hasattr(route, "methods")
        }
        for path in ("/health", "/stats", "/phones", "/chat", "/compare", "/reviews"):
            assert path in page, f"UI never calls {path}"
            assert path in served, f"UI calls undocumented {path}"

    def test_health_reports_ready(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        assert body["database"] is True
        assert body["phones_indexed"] >= 10
        assert body["documents_indexed"] > 50

    def test_stats(self, client):
        body = client.get("/stats").json()
        assert body["phones"] >= 10
        assert body["specifications"] > 100
        assert body["release_years"]

    def test_openapi_schema_is_served(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert "/chat" in response.json()["paths"]


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------
class TestPhoneEndpoints:
    def test_list_phones(self, client):
        body = client.get("/phones").json()
        assert body["total"] >= 10
        assert len(body["phones"]) == body["count"]
        assert body["phones"][0]["name"]

    def test_list_phones_pagination(self, client):
        first = client.get("/phones?limit=3&offset=0").json()
        second = client.get("/phones?limit=3&offset=3").json()

        assert len(first["phones"]) == 3
        first_ids = {p["id"] for p in first["phones"]}
        second_ids = {p["id"] for p in second["phones"]}
        assert first_ids.isdisjoint(second_ids)

    def test_pagination_rejects_invalid_limit(self, client):
        assert client.get("/phones?limit=0").status_code == 422
        assert client.get("/phones?limit=500").status_code == 422

    def test_search(self, client):
        results = client.get("/phones/search?q=Ultra").json()
        assert results
        assert all("Ultra" in phone["name"] for phone in results)

    def test_search_requires_a_query(self, client):
        assert client.get("/phones/search?q=").status_code == 422

    def test_get_phone_by_slug(self, client):
        body = client.get("/phones/samsung-galaxy-s23").json()
        assert body["name"] == "Samsung Galaxy S23"
        assert body["battery_capacity_mah"] == 3900
        assert body["specifications"]

    def test_get_phone_by_id(self, client):
        listed = client.get("/phones?limit=1").json()["phones"][0]
        body = client.get(f"/phones/{listed['id']}").json()
        assert body["id"] == listed["id"]

    def test_get_phone_by_loose_name(self, client):
        body = client.get("/phones/Galaxy S23 Ultra").json()
        assert "S23 Ultra" in body["name"]

    def test_unknown_phone_returns_404(self, client):
        assert client.get("/phones/nokia-3310").status_code == 404

    def test_plain_text_specifications(self, client):
        response = client.get("/phones/samsung-galaxy-s23/specifications")
        assert response.status_code == 200
        assert "Display" in response.text


# ---------------------------------------------------------------------------
# Comparison (no language model involved)
# ---------------------------------------------------------------------------
class TestCompareEndpoint:
    def test_compares_two_phones(self, client):
        response = client.post(
            "/compare", json={"phones": ["Galaxy S24 Ultra", "Galaxy S23 Ultra"]}
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["phones"]) == 2
        assert body["rows"]

    def test_winner_is_marked_only_where_values_differ(self, client):
        body = client.post(
            "/compare", json={"phones": ["Galaxy S24 Ultra", "Galaxy S23 Ultra"]}
        ).json()

        for row in body["rows"]:
            values = set(row["values"].values())
            if row["winner"] is not None:
                assert len(values) > 1, row["attribute"]

    def test_released_year_winner_is_the_newer_phone(self, client):
        body = client.post(
            "/compare", json={"phones": ["Galaxy S24 Ultra", "Galaxy S23 Ultra"]}
        ).json()
        released = next(r for r in body["rows"] if r["attribute"] == "Released")
        assert released["winner"] == "Samsung Galaxy S24 Ultra"

    def test_requires_at_least_two_phones(self, client):
        assert client.post("/compare", json={"phones": ["Galaxy S23"]}).status_code == 422

    def test_unknown_phone_returns_404(self, client):
        response = client.post(
            "/compare", json={"phones": ["Galaxy S23", "Nokia 3310"]}
        )
        assert response.status_code == 404

    def test_same_phone_twice_is_rejected(self, client):
        response = client.post(
            "/compare", json={"phones": ["Galaxy S23", "galaxy s23"]}
        )
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
class TestAgentEndpoints:
    def test_agents_are_described(self, client):
        agents = client.get("/agents").json()
        assert len(agents) == 3
        names = [agent["name"] for agent in agents]
        assert names == [
            "specification_agent",
            "comparison_agent",
            "review_agent",
        ]

    def test_review_of_unknown_phone_returns_404(self, client):
        response = client.post("/reviews", json={"phone": "Nokia 3310"})
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Chat and review generation (these run the model)
# ---------------------------------------------------------------------------
class TestChatEndpoint:
    def test_empty_question_is_rejected(self, client):
        assert client.post("/chat", json={"question": ""}).status_code == 422

    def test_overlong_question_is_rejected(self, client):
        response = client.post("/chat", json={"question": "x" * 600})
        assert response.status_code == 422

    @pytest.mark.llm
    def test_specification_question(self, client):
        body = client.post(
            "/chat", json={"question": "What is the screen size of the Galaxy S22?"}
        ).json()

        assert "6.1" in body["answer"]
        assert body["intent"] == "spec_lookup"
        assert body["sources"]
        assert body["elapsed_seconds"] > 0

    @pytest.mark.llm
    def test_superlative_question(self, client):
        body = client.post(
            "/chat", json={"question": "Which Samsung phone has the best battery life?"}
        ).json()
        assert body["intent"] == "superlative"
        assert "S25 Ultra" in body["answer"]

    @pytest.mark.llm
    def test_context_is_returned_when_requested(self, client):
        body = client.post(
            "/chat",
            json={
                "question": "What is the battery of the Galaxy S23?",
                "include_context": True,
            },
        ).json()
        assert body["context"]


@pytest.mark.llm
class TestReviewEndpoint:
    def test_generates_a_full_review(self, client):
        response = client.post("/reviews", json={"phone": "Galaxy A54"})
        assert response.status_code == 200

        body = response.json()
        assert body["success"] is True
        assert len(body["agents"]) == 3
        assert all(step["success"] for step in body["agents"])

        review = body["review"]
        assert len(review["sections"]) == 4
        assert review["markdown"].startswith("# ")
        assert review["verdict"]
