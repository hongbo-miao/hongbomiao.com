import time
from datetime import datetime

import sentry_sdk
from flask import Flask, request
from flask_apscheduler import APScheduler
from flask_cors import CORS
from flask_sock import Sock
from sentry_sdk.integrations.flask import FlaskIntegration
from simple_websocket import Server as WebSocketServer

lucky_number = 0


def create_app() -> Flask:
    sentry_sdk.init(
        dsn="https://a6fe08e6f20f4eb7929b85a513c39dfa@o379185.ingest.sentry.io/4504195631480832",
        integrations=[FlaskIntegration()],
        traces_sample_rate=1.0,
    )

    app = Flask(__name__)
    CORS(app)
    sock = Sock(app)
    scheduler = APScheduler()
    scheduler.init_app(app)
    scheduler.start()

    @scheduler.task(
        "interval",
        id="fetch_lucky_number",
        seconds=5,
        misfire_grace_time=10,
        max_instances=1,
    )
    def fetch_lucky_number():
        time.sleep(3)
        global lucky_number
        lucky_number += 1

    @app.route("/")
    def get_health() -> dict[str, str]:
        return {
            "api": "ok",
        }

    @app.get("/seed")
    def get_seed() -> dict[str, int]:
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

    @app.post("/update_lucky_number")
    def update_lucky_number() -> dict[str, int]:
        scheduler.pause()
        time.sleep(1)
        global lucky_number
        lucky_number = 0
        scheduler.resume()

        # Trigger a new status update immediately
        for job in scheduler.get_jobs():
            job.modify(next_run_time=datetime.now())
        return {
            "luckyNumber": lucky_number,
        }

    @sock.route("/lucky-number")
    def get_lucky_number(ws: WebSocketServer) -> None:
        while True:
            ws.send(lucky_number)
            time.sleep(1)

    return app
