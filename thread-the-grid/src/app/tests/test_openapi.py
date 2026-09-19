# src/app/tests/test_openapi.py
"""Swagger groups endpoints by tag, so an endpoint with two tags is listed twice."""

from fastapi.testclient import TestClient

from src.app.main import app

client = TestClient(app)


def test_every_endpoint_is_listed_under_exactly_one_tag() -> None:
    spec = client.get("/openapi.json").json()
    tags = {
        f"{method.upper()} {path}": operation.get("tags", [])
        for path, operations in spec["paths"].items()
        for method, operation in operations.items()
    }
    assert {endpoint: t for endpoint, t in tags.items() if len(t) != 1} == {}
