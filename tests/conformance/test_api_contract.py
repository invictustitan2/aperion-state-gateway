"""API contract conformance tests for Cortex endpoints.

These tests verify that the API endpoints conform to the contract
defined in Aperion For Dummies (Chapter 19).
"""

import pytest
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from aperion_cortex.service.app import create_cortex_app
from aperion_cortex.service.config import CortexConfig, ScopedToken
from aperion_cortex.service.models import Resource, Database, Service, ServiceHealth


@pytest.fixture
def config():
    """Create test configuration."""
    config = CortexConfig(signing_key="test-signing-key-12345")
    config.tokens = [
        ScopedToken(token="test-token", scopes=["*"]),
    ]
    return config


@pytest.fixture
def app(config):
    """Create test application."""
    # Create mock providers
    def resource_provider():
        return [
            Resource(
                id="db-primary",
                kind="database",
                name="Primary Database",
                uri="postgresql://...",
                metadata={},
            )
        ]

    def hot_fact_provider():
        return ["Database healthy", "FSAL service running"]

    def summary_provider(resources):
        return f"System snapshot with {len(resources)} resources"

    def database_provider():
        return [
            Database(
                name="primary",
                engine="postgresql",
                dsn_ref="env:DATABASE_URL",
                table_count=15,
            )
        ]

    def service_provider():
        return [
            Service(
                name="fsal",
                kind="filesystem",
                endpoint="http://localhost:4848",
            ),
            Service(
                name="cortex",
                kind="api",
                endpoint="http://localhost:4949",
            ),
        ]

    def service_health_provider():
        return [
            ServiceHealth(
                name="cortex",
                status="healthy",
                latency_ms=5,
                last_checked=datetime.now(timezone.utc),
            )
        ]

    return create_cortex_app(
        config,
        resource_provider=resource_provider,
        hot_fact_provider=hot_fact_provider,
        summary_provider=summary_provider,
        database_provider=database_provider,
        service_provider=service_provider,
        service_health_provider=service_health_provider,
    )


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Create authorization headers."""
    return {"Authorization": "Bearer test-token"}


class TestHealthEndpoints:
    """Tests for health check endpoints (no auth required)."""

    def test_healthz_returns_ok(self, client):
        """GET /healthz should return ok status."""
        response = client.get("/healthz")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "cortex"
        assert "version" in data

    def test_ready_returns_ready(self, client):
        """GET /ready should return ready status."""
        response = client.get("/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True


class TestStateEndpoints:
    """Tests for state management endpoints."""

    def test_state_discover_requires_auth(self, client):
        """GET /state/discover should require authentication."""
        response = client.get("/state/discover")
        assert response.status_code == 401

    def test_state_discover_returns_resources(self, client, auth_headers):
        """GET /state/discover should return resources."""
        response = client.get("/state/discover", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "resources" in data
        assert len(data["resources"]) > 0
        assert data["resources"][0]["id"] == "db-primary"

    def test_state_snapshot_returns_signed_pack(self, client, auth_headers):
        """GET /state/snapshot should return signed ContextPack."""
        response = client.get("/state/snapshot", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()

        # Verify ContextPack schema
        assert "generated_at" in data
        assert "ttl_seconds" in data
        assert "summary" in data
        assert "resources" in data
        assert "hot_facts" in data
        assert "signature" in data
        assert "freshness" in data

    def test_state_snapshot_with_since_parameter(self, client, auth_headers):
        """GET /state/snapshot?since should respect since parameter."""
        # First request to get a timestamp
        response1 = client.get("/state/snapshot", headers=auth_headers)
        assert response1.status_code == 200
        data1 = response1.json()

        # Second request with since from first
        since = data1["generated_at"]
        response2 = client.get(
            f"/state/snapshot?since={since}",
            headers=auth_headers,
        )

        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["freshness"] in ["cached", "fresh"]


class TestVectorEndpoints:
    """Tests for vector search endpoints."""

    def test_vector_search_requires_auth(self, client):
        """POST /vector/search should require authentication."""
        response = client.post("/vector/search", json={"query": "test"})
        assert response.status_code == 401

    def test_vector_search_returns_hits(self, client, auth_headers):
        """POST /vector/search should return hits array."""
        response = client.post(
            "/vector/search",
            json={"query": "authentication", "top_k": 5},
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "hits" in data
        assert isinstance(data["hits"], list)


class TestDatabaseEndpoints:
    """Tests for database introspection endpoints."""

    def test_db_list_returns_databases(self, client, auth_headers):
        """GET /db/list should return database list."""
        response = client.get("/db/list", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert data[0]["name"] == "primary"

    def test_db_schema_returns_schema(self, client, auth_headers):
        """GET /db/schema/{name} should return schema."""
        response = client.get("/db/schema/primary", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "tables" in data
        assert "schema_hash" in data
        assert "computed_at" in data


class TestServiceEndpoints:
    """Tests for service management endpoints."""

    def test_svc_list_returns_services(self, client, auth_headers):
        """GET /svc/list should return services."""
        response = client.get("/svc/list", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 2

    def test_svc_health_returns_health(self, client, auth_headers):
        """GET /svc/health should return health status."""
        response = client.get("/svc/health", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert data[0]["status"] in ["healthy", "ok", "degraded", "unhealthy"]


class TestAuditEndpoints:
    """Tests for audit logging endpoints."""

    def test_audit_log_returns_entries(self, client, auth_headers):
        """GET /audit/log should return entries."""
        response = client.get("/audit/log", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "entries" in data

    def test_audit_record_creates_entry(self, client, auth_headers):
        """POST /audit/record should create ledger entry."""
        response = client.post(
            "/audit/record",
            json={
                "run_id": "test-run-123",
                "agent_name": "TestAgent",
                "method_name": "test_method",
                "prompt": "Test prompt",
                "raw_output": "Test output",
            },
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "run_id" in data
        assert "recorded_at" in data
        assert "ledger_path" in data


class TestContextOptimization:
    """Tests for context optimization endpoints."""

    def test_context_optimize_returns_packed_items(self, client, auth_headers):
        """POST /context/optimize should return optimized items."""
        response = client.post(
            "/context/optimize",
            json={
                "items": [
                    {"content": "High priority item", "priority": 10},
                    {"content": "Low priority item", "priority": 1},
                ],
                "max_tokens": 1000,
            },
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total_tokens" in data
        assert "items_included" in data
        assert "items_dropped" in data
        assert "budget_used_percent" in data


class TestAuthenticationScopes:
    """Tests for authentication scope enforcement."""

    def test_missing_token_returns_401(self, client):
        """Missing token should return 401."""
        response = client.get("/state/discover")
        assert response.status_code == 401

    def test_invalid_token_returns_401(self, client):
        """Invalid token should return 401."""
        response = client.get(
            "/state/discover",
            headers={"Authorization": "Bearer invalid-token"},
        )
        assert response.status_code == 401

    def test_query_param_token_works(self, client):
        """Token can be passed as query parameter (for SSE)."""
        response = client.get("/state/discover?token=test-token")
        assert response.status_code == 200
