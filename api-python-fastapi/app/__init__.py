from fastapi import FastAPI, WebSocket
from pydantic import BaseModel


class Seed(BaseModel):
    seedNumber: int


seed_number = 42
app = FastAPI()


@app.get("/")
async def get_health() -> dict[str, str]:
    return {"api": "ok"}


@app.get("/seed")
async def get_seed() -> dict[str, int]:
    return {"seedNumber": seed_number}


@app.post("/update-seed")
async def update_seed(seed: Seed) -> dict[str, int]:
    global seed_number
    seed_number = seed.seedNumber
    return {"seedNumber": seed_number}


@app.websocket("/echo")
async def echo(ws: WebSocket):
    await ws.accept()
    while True:
        data = await ws.receive_text()
        await ws.send_text(data)
