import os

from pydantic_settings import BaseSettings

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def get_env_file() -> str:
    env = os.getenv("ENV")
    return ".env.production" if env == "production" else ".env.development"


class Config(BaseSettings):
    TOKENIZERS_PARALLELISM: bool
    OPENAI_API_KEY: str
    model_config = {
        "env_file": get_env_file(),
    }
