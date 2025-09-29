from app import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_get_health() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"api": "ok"}
