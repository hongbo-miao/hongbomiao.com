import os

from pydantic_settings import BaseSettings, SettingsConfigDict


def get_environment_files() -> list[str]:
    environment = os.getenv("ENVIRONMENT")
    match environment:
        case "development" | "test":
            return [".env.development", ".env.development.local"]
        case "production":
            return [".env.production", ".env.production.local"]
        case _:
            message = f"Invalid ENVIRONMENT value: {environment}."
            raise ValueError(message)


class Config(BaseSettings):
    AWS_ACCESS_KEY_ID: str
    AWS_DEFAULT_REGION: str
    AWS_SECRET_ACCESS_KEY: str
    PARQUET_PATH: str | None = None
    S3_PATH: str

    model_config = SettingsConfigDict(env_file=get_environment_files())


config = Config.model_validate({})
