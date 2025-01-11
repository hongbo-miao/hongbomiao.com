import os

from pydantic_settings import BaseSettings


def get_env_file() -> str:
    env = os.getenv("ENV")
    return ".env.production.local" if env == "production" else ".env.development.local"


class Config(BaseSettings):
    HTTP_AUTH_TOKEN: str

    model_config = {
        "env_file": get_env_file(),
    }
