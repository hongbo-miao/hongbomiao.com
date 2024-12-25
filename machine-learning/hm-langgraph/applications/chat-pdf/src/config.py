import os

from pydantic_settings import BaseSettings


def get_env_file() -> str:
    env = os.getenv("ENV")
    return ".env.production" if env == "production" else ".env.development"


class Config(BaseSettings):
    OPENAI_API_KEY: str
    TOKENIZERS_PARALLELISM: bool
    model_config = {
        "env_file": get_env_file(),
    }
