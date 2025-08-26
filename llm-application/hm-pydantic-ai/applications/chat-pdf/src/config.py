import os

from pydantic_settings import BaseSettings, SettingsConfigDict

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def get_env_files() -> list[str]:
    env = os.getenv("ENV")
    if env == "production":
        return [".env.production", ".env.production.local"]
    return [".env.development", ".env.development.local"]


class Config(BaseSettings):
    TOKENIZERS_PARALLELISM: bool
    OPENAI_API_KEY: str

    model_config = SettingsConfigDict(env_file=get_env_files())


config = Config.model_validate({})
