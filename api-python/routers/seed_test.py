from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get_seed():
    response = client.get("/seed")
    assert response.status_code == 200
    assert response.json() == {"seedNumber": 42}


def test_update_seed():
    new_seed = 100
    response = client.post("/update-seed", json={"seedNumber": new_seed})
    assert response.status_code == 200
    assert response.json() == {"seedNumber": new_seed}

    response = client.get("/seed")
    assert response.status_code == 200
    assert response.json() == {"seedNumber": new_seed}
