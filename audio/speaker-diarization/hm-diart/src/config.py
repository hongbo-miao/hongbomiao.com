import os

from pydantic_settings import BaseSettings, SettingsConfigDict


def get_env_files() -> list[str]:
    env = os.getenv("ENV")
    if env == "production":
        return [".env.production", ".env.production.local"]
    if env in {"development", "test"}:
        return [".env.development", ".env.development.local"]
    message = f"Invalid ENV value: {env}."
    raise ValueError(message)


class Config(BaseSettings):
    ENV: str
    HUGGING_FACE_HUB_TOKEN: str
    SEGMENTATION_MODEL: str
    EMBEDDING_MODEL: str

    model_config = SettingsConfigDict(env_file=get_env_files())


config = Config.model_validate({})
