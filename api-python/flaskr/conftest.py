import pytest
from flask import Flask
from flask.testing import FlaskClient, FlaskCliRunner
from flaskr import create_app


@pytest.fixture()
def app() -> Flask:
    app = create_app()
    app.config.update({"TESTING": True})
    yield app


@pytest.fixture()
def client(app) -> FlaskClient:
    return app.test_client()


@pytest.fixture()
def runner(app) -> FlaskCliRunner:
    return app.test_cli_runner()
