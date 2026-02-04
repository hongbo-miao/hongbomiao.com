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
    MOVEMENT_VELOCITY_THRESHOLD_MPS: float
    NUSCENES_DATASET_DIRECTORY_PATH: str
    NUSCENES_VERSION: str
    NUSCENES_SCENE_INDEX: int
    VISUALIZATION_FRAME_COUNT: int
    YOLO_MODEL_PATH: str
    CAMERA_CONFIDENCE_WEIGHT: float = 0.7
    FUSION_BASE_CONFIDENCE: float = 0.3

    model_config = SettingsConfigDict(env_file=get_environment_files())


config = Config.model_validate({})
