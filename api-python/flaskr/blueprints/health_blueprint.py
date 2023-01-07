from flask import Blueprint

health_blueprint = Blueprint("health_blueprint", __name__)


@health_blueprint.route("/")
def get_health() -> dict[str, str]:
    return {"api": "ok"}
