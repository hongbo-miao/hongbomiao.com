from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/")
async def get_health() -> dict[str, str]:
    return {"api": "ok"}
