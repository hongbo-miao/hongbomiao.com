import os

from pydantic_settings import BaseSettings


def get_env_file() -> list[str]:
    env = os.getenv("ENV")
    if env == "production":
        return [".env.production", ".env.production.local"]
    return [".env.development", ".env.development.local"]


class Config(BaseSettings):
    ENV: str
    SENTRY_DSN: str
    KAFKA_BOOTSTRAP_SERVERS: str

    model_config = {
        "env_file": get_env_file(),
    }


config = Config()
