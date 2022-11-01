from flask import Flask
from flask_sock import Sock

app = Flask(__name__)
sock = Sock(app)


@app.route("/")
def hello():
    return "Hello World!"


@sock.route("/echo")
def echo(ws):
    while True:
        data = ws.receive()
        ws.send(data)
