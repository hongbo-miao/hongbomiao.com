import os

from pydantic_settings import BaseSettings


def get_env_file() -> str:
    env = os.getenv("ENV")
    return ".env.production.local" if env == "production" else ".env.development.local"


class Config(BaseSettings):
    ENV: str
    FIRECRAWL_API_KEY: str

    model_config = {
        "env_file": get_env_file(),
    }


config = Config()
