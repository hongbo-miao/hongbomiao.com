import random
import time

from flask import Flask, request
from flask_cors import CORS
from flask_sock import Sock
from simple_websocket import Server as WebSocketServer


def create_app() -> Flask:
    app = Flask(__name__)

    sock = Sock(app)
    CORS(app)

    @app.route("/")
    def health() -> str:
        return "ok"

    @app.get("/seed")
    def seed() -> dict[str, int]:
        return {
            "seedNumber": 42,
        }

    @app.post("/update_seed")
    def update_seed() -> dict[str, int]:
        return {
            "seedNumber": request.json["seedNumber"],
        }

    @sock.route("/echo")
    def echo(ws: WebSocketServer) -> None:
        while True:
            data = ws.receive()
            ws.send(data)

    @sock.route("/lucky-number")
    def num(ws: WebSocketServer) -> None:
        while True:
            n = random.random()
            ws.send(n)
            time.sleep(1)

    return app
