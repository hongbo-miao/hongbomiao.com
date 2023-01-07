import json
import logging
import random
import time
from datetime import datetime

import sentry_sdk
from confluent_kafka import Producer
from flask import Flask, request
from flask_apscheduler import APScheduler
from flask_cors import CORS
from flask_sock import Sock
from flaskr.utils import kafka_util
from sentry_sdk.integrations.flask import FlaskIntegration
from simple_websocket import Server as WebSocketServer

seed_number = 42
lucky_number = 0
lucky_number_client_count = 0


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_pyfile("config.py")

    if __name__ != "__main__":
        gunicorn_logger = logging.getLogger("gunicorn.error")
        if gunicorn_logger.level != logging.NOTSET:
            app.logger.handlers = gunicorn_logger.handlers
            app.logger.setLevel(gunicorn_logger.level)

    sentry_sdk.init(
        dsn=app.config.get("SENTRY_DSN"),
        integrations=[FlaskIntegration()],
        traces_sample_rate=1.0,
        environment=app.config.get("ENV"),
    )
    CORS(app)
    sock = Sock(app)
    scheduler = APScheduler()
    scheduler.init_app(app)
    scheduler.start(paused=True)

    @app.cli.command("greet")
    def greet():
        app.logger.info(f"{datetime.utcnow()} Hello")
        time.sleep(3)
        app.logger.info(f"{datetime.utcnow()} Bye")

    @scheduler.task(
        "interval",
        id="fetch_lucky_number",
        seconds=5,
        misfire_grace_time=10,
        max_instances=1,
    )
    def fetch_lucky_number():
        global lucky_number
        app.logger.info(f"lucky_number: {lucky_number}")
        lucky_number += 1

    @app.route("/")
    def get_health() -> dict[str, str]:
        return {"api": "ok"}

    @app.get("/seed")
    def get_seed() -> dict[str, int]:
        return {"seedNumber": seed_number}

    @app.post("/update-seed")
    def update_seed() -> dict[str, int]:
        global seed_number
        seed_number = request.json["seedNumber"]
        return {"seedNumber": seed_number}

    @app.post("/generate-garden-sensor-data")
    def generate_garden_sensor_data() -> dict[str, bool]:
        producer = Producer(
            {"bootstrap.servers": app.config.get("KAFKA_BOOTSTRAP_SERVERS")}
        )
        for _ in range(5):
            data = {
                "temperature": round(random.uniform(-10, 50), 1),
                "humidity": round(random.uniform(0, 100), 1),
                "wind": round(random.uniform(0, 10), 1),
                "soil": round(random.uniform(0, 100), 1),
            }
            producer.poll(0)
            producer.produce(
                "garden_sensor_data",
                json.dumps(data).encode("utf-8"),
                callback=kafka_util.delivery_report,
            )
            time.sleep(1)
        producer.flush()
        return {"body": True}

    @sock.route("/echo")
    def echo(ws: WebSocketServer) -> None:
        while True:
            data = ws.receive()
            ws.send(data)

    @app.post("/update-lucky-number")
    def update_lucky_number() -> dict[str, int]:
        global lucky_number
        try:
            scheduler.pause()
            lucky_number = 0
        except Exception as e:
            app.logger.error(e)
        else:
            now = datetime.now()
            for job in scheduler.get_jobs():
                job.modify(next_run_time=now)
        finally:
            scheduler.resume()
            return {"luckyNumber": lucky_number}

    @sock.route("/lucky-number")
    def get_lucky_number(ws: WebSocketServer) -> None:
        global lucky_number_client_count
        lucky_number_client_count += 1
        app.logger.info(f"lucky_number_client_count: {lucky_number_client_count}")
        if lucky_number_client_count > 0:
            scheduler.resume()
        try:
            while True:
                ws.send(lucky_number)
                time.sleep(1)
        finally:
            lucky_number_client_count -= 1
            app.logger.info(f"lucky_number_client_count: {lucky_number_client_count}")
            if lucky_number_client_count == 0:
                scheduler.pause()

    return app
