from flask import Flask
from flask_sock import Sock
from flask_wtf.csrf import CSRFProtect
from simple_websocket import Server as WebSocketServer


def create_app() -> Flask:
    app = Flask(__name__)
    sock = Sock(app)
    csrf = CSRFProtect()
    csrf.init_app(app)

    @app.route("/")
    def health() -> str:
        return "ok"

    @sock.route("/echo")
    def echo(ws: WebSocketServer) -> None:
        while True:
            data = ws.receive()
            ws.send(data)

    return app
