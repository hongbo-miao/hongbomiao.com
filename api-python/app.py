import time

from flask import Flask, request
from flask_apscheduler import APScheduler
from flask_cors import CORS
from flask_sock import Sock
from simple_websocket import Server as WebSocketServer

lucky_number = 0


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)
    sock = Sock(app)
    scheduler = APScheduler(app)
    scheduler.start()

    @scheduler.task(
        "interval", id="increase_lucky_number", seconds=1, misfire_grace_time=10
    )
    def increase_lucky_number():
        global lucky_number
        lucky_number += 1

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
            ws.send(lucky_number)
            time.sleep(3)

    return app
