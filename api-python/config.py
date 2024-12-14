from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ENV: str = "development"
    SENTRY_DSN: str | None = None
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
