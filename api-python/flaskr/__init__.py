import logging
import time
from datetime import datetime

import sentry_sdk
from flask import Flask
from flask_cors import CORS
from flaskr.blueprints.health_blueprint import health_blueprint
from flaskr.blueprints.motor_blueprint import motor_blueprint
from flaskr.blueprints.seed_blueprint import seed_blueprint
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
    app.register_blueprint(health_blueprint)
    app.register_blueprint(motor_blueprint)
    app.register_blueprint(seed_blueprint)

    @app.cli.command("greet")
    def greet():
        app.logger.info(f"{datetime.utcnow()} Hello")
        time.sleep(2)
        app.logger.info(f"{datetime.utcnow()} Bye")

    return app
