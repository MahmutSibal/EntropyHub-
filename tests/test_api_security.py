from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)
VALID_KEY = "teknofest-local-dev-key"


def test_protected_endpoint_requires_api_key():
    response = client.post("/random/bytes", json={"count": 4})
    assert response.status_code == 401
    assert "x-api-key" in response.json()["detail"]


def test_rate_limit_returns_retry_after_header():
    limiter = app.state.rate_limiter
    limiter.reset()
    original_limit = limiter.limit_per_minute
    limiter.limit_per_minute = 1

    try:
        first = client.post(
            "/random/bytes",
            json={"count": 4},
            headers={"x-api-key": VALID_KEY},
        )
        second = client.post(
            "/random/bytes",
            json={"count": 4},
            headers={"x-api-key": VALID_KEY},
        )

        assert first.status_code == 200
        assert second.status_code == 429
        assert second.headers.get("Retry-After") is not None
    finally:
        limiter.limit_per_minute = original_limit
        limiter.reset()


def test_docs_path_is_public():
    response = client.get("/docs")
    assert response.status_code == 200


def test_metrics_path_is_public():
    response = client.get("/metrics")
    assert response.status_code == 200
