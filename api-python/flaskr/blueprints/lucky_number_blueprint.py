import time

from flask import Blueprint, abort, current_app
from flaskr.utils.scheduler import scheduler
from flaskr.utils.sock import sock
from simple_websocket import Server as WebSocketServer

lucky_number = 0
lucky_number_client_count = 0
lucky_number_blueprint = Blueprint("lucky_number_blueprint", __name__)


@scheduler.task(
    "interval",
    id="fetch_lucky_number",
    seconds=5,
    misfire_grace_time=10,
    max_instances=1,
)
def fetch_lucky_number():
    global lucky_number
    lucky_number += 1


@lucky_number_blueprint.post("/update-lucky-number")
def update_lucky_number() -> dict[str, int]:
    global lucky_number
    try:
        scheduler.pause()
        lucky_number = 0
    except Exception as e:
        current_app.logger.error(e)
        abort(500, e)
    finally:
        scheduler.resume()
    return {"luckyNumber": lucky_number}


@sock.route("/lucky-number")
def get_lucky_number(ws: WebSocketServer) -> None:
    global lucky_number_client_count
    lucky_number_client_count += 1
    current_app.logger.info(f"lucky_number_client_count: {lucky_number_client_count}")
    if lucky_number_client_count > 0:
        scheduler.resume()
    try:
        while True:
            ws.send(lucky_number)
            time.sleep(1)
    finally:
        lucky_number_client_count -= 1
        current_app.logger.info(
            f"lucky_number_client_count: {lucky_number_client_count}"
        )
        if lucky_number_client_count == 0:
            scheduler.pause()
