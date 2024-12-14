import os

from pydantic_settings import BaseSettings


def get_env_file() -> str:
    env = os.getenv("ENV")
    return ".env.production" if env == "production" else ".env.development"


class Settings(BaseSettings):
    ENV: str
    SENTRY_DSN: str
    KAFKA_BOOTSTRAP_SERVERS: str

    model_config = {
        "env_file": get_env_file(),
    }
