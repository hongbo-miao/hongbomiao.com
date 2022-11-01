from flask import Flask
from flask_sock import Sock
from flask_wtf.csrf import CSRFProtect


def create_app() -> Flask:
    app = Flask(__name__)
    sock = Sock(app)
    csrf = CSRFProtect()
    csrf.init_app(app)

    @app.route("/")
    def hello():
        return "Hello World!"

    @sock.route("/echo")
    def echo(ws):
        while True:
            data = ws.receive()
            ws.send(data)

    return app
