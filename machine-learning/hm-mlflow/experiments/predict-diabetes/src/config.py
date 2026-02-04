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
    MLFLOW_EXPERIMENT_NAME: str
    MLFLOW_TRACKING_SERVER_HOST: str
    MLFLOW_TRACKING_USERNAME: str
    MLFLOW_TRACKING_PASSWORD: str

    model_config = SettingsConfigDict(env_file=get_environment_files())


config = Config.model_validate({})
