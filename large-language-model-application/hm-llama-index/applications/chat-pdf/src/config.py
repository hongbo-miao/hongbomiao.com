import os

from pydantic_settings import BaseSettings, SettingsConfigDict

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def get_env_files() -> list[str]:
    env = os.getenv("ENV")
    match env:
        case "production":
            return [".env.production", ".env.production.local"]
        case "development" | "test":
            return [".env.development", ".env.development.local"]
        case _:
            message = f"Invalid ENV value: {env}."
            raise ValueError(message)


class Config(BaseSettings):
    TOKENIZERS_PARALLELISM: bool
    OPENAI_API_KEY: str

    model_config = SettingsConfigDict(env_file=get_env_files())


config = Config.model_validate({})
