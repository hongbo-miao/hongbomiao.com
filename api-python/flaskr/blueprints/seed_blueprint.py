from flask import Blueprint, request

seed_number = 42
seed_blueprint = Blueprint("seed_blueprint", __name__)


@seed_blueprint.get("/seed")
def get_seed() -> dict[str, int]:
    return {"seedNumber": seed_number}


@seed_blueprint.post("/update-seed")
def update_seed() -> dict[str, int]:
    global seed_number
    seed_number = request.json["seedNumber"]
    return {"seedNumber": seed_number}
