from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["seed"])

seed_number = 42


class SeedUpdate(BaseModel):
    seedNumber: int

    class Config:
        json_schema_extra = {"example": {"seedNumber": 100}}


@router.get("/seed")
async def get_seed() -> dict[str, int]:
    return {"seedNumber": seed_number}


@router.post("/update-seed")
async def update_seed(seed_data: SeedUpdate) -> dict[str, int]:
    global seed_number
    seed_number = seed_data.seedNumber
    return {"seedNumber": seed_number}
