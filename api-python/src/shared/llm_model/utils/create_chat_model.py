from config import config
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


def create_chat_model() -> OpenAIChatModel:
    return OpenAIChatModel(
        model_name=config.CHAT_MODEL,
        provider=OpenAIProvider(
            base_url=f"{config.OPENAI_API_BASE_URL}/v1",
            api_key=config.OPENAI_API_KEY,
        ),
    )
