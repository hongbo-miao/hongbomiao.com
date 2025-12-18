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
    AWS_ACCESS_KEY_ID: str
    AWS_DEFAULT_REGION: str
    AWS_SECRET_ACCESS_KEY: str
    PARQUET_PATH: str | None = None
    S3_PATH: str

    model_config = SettingsConfigDict(env_file=get_env_files())


config = Config.model_validate({})
