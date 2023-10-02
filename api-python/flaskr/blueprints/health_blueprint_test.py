from flask.testing import FlaskClient


class TestHealthBlueprint:
    def test_request_example(self, client: FlaskClient):
        res = client.get("/")
        assert res.status_code == 200
        assert res.json["api"] == "ok"
