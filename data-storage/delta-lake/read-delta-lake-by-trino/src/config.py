import os

from pydantic_settings import BaseSettings, SettingsConfigDict


def get_env_files() -> list[str]:
    env = os.getenv("ENV")
    if env is None:
        env = "production"
    match env:
        case "production":
            return [".env.production", ".env.production.local"]
        case "development" | "test":
            return [".env.development", ".env.development.local"]
        case _:
            message = f"Invalid ENV value: {env}."
            raise ValueError(message)


class Config(BaseSettings):
    TRINO_HOST: str
    TRINO_PORT: int
    TRINO_USER: str

    model_config = SettingsConfigDict(env_file=get_env_files())


config = Config.model_validate({})
