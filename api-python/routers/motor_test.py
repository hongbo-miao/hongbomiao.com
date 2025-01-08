from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


@patch("routers.motor.Producer")
def test_generate_motor_data(mock_producer: MagicMock) -> None:
    mock_producer_instance = MagicMock()
    mock_producer.return_value = mock_producer_instance

    response = client.post("/generate-motor-data")
    assert response.status_code == 200
    assert response.json() == {"body": True}

    assert mock_producer_instance.produce.call_count == 5
    assert mock_producer_instance.poll.call_count == 5
