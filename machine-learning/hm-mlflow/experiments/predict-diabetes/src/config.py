import os

from dotenv import load_dotenv

load_dotenv(".env.production.local")

MLFLOW_TRACKING_SERVER_URL = os.getenv("MLFLOW_TRACKING_SERVER_URL")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")
