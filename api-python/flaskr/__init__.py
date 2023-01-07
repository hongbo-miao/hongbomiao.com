import json
import logging
import random
import time
from datetime import datetime

import sentry_sdk
from confluent_kafka import Producer
from flask import Flask
from flask_cors import CORS
from flaskr.blueprints.health_blueprint import health_blueprint
from flaskr.blueprints.lucky_number_blueprint import lucky_number_blueprint
from flaskr.blueprints.seed_blueprint import seed_blueprint
from flaskr.utils import kafka_util
from flaskr.utils.scheduler import scheduler
from flaskr.utils.sock import sock
from sentry_sdk.integrations.flask import FlaskIntegration


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
    sock.init_app(app)
    scheduler.init_app(app)
    app.register_blueprint(health_blueprint)
    app.register_blueprint(seed_blueprint)
    app.register_blueprint(lucky_number_blueprint)

    scheduler.start()

    @app.cli.command("greet")
    def greet():
        app.logger.info(f"{datetime.utcnow()} Hello")
        time.sleep(2)
        app.logger.info(f"{datetime.utcnow()} Bye")

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

    return app
