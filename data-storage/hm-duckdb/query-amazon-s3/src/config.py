import os

from pydantic_settings import BaseSettings, SettingsConfigDict


def get_env_files() -> list[str]:
    env = os.getenv("ENV")
    if env == "production":
        return [".env.production", ".env.production.local"]
    return [".env.development", ".env.development.local"]


class Config(BaseSettings):
    HTTP_AUTH_TOKEN: str

    model_config = SettingsConfigDict(env_file=get_env_files())


config = Config.model_validate({})
