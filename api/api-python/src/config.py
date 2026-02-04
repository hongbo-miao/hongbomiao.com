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
    ENVIRONMENT: str
    SERVER_HOST: str
    SERVER_PORT: int
    SERVER_RELOAD: bool
    SENTRY_DSN: str
    DOCUMENT_LANCE_DB_DIR: str
    EMBEDDING_MODEL: str
    CHAT_MODEL: str
    CHAT_MODEL_TEMPERATURE: float
    CHAT_MODEL_MAX_TOKENS: int
    MEMORY_MODEL: str
    MEMORY_LIMIT: int
    MEMORY_MODEL_TEMPERATURE: float
    MEMORY_MODEL_MAX_TOKENS: int
    OPENAI_API_BASE_URL: str
    OPENAI_API_KEY: str | None = None
    KAFKA_BOOTSTRAP_SERVERS: str

    model_config = SettingsConfigDict(env_file=get_environment_files())


config = Config.model_validate({})
