import time

from config import config
from fastapi import APIRouter
from shared.openai.types.model_info import ModelInfo
from shared.openai.types.models_response import ModelsResponse

router = APIRouter(tags=["models"])


@router.get("/v1/models")
async def list_models() -> ModelsResponse:
    model_id = config.CHAT_MODEL
    return ModelsResponse(
        data=[
            ModelInfo(
                id=model_id,
                created=int(time.time()),
                owned_by="hongbo-miao",
            ),
        ],
    )
