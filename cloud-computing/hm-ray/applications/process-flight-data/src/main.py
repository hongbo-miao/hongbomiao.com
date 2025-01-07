import logging
import tempfile
from pathlib import Path

import config
import mlflow
import pandas as pd
import ray

logger = logging.getLogger(__name__)


@ray.remote
def process_flight_data(
    mlflow_tracking_server_host: str,
    mlflow_tracking_server_user_name: str,
    mlflow_tracking_server_password: str,
    mlflow_experiment_name: str,
    flight_data: dict,
    experiment_number: int,
) -> pd.DataFrame:
    mlflow.set_tracking_uri(
        f"https://{mlflow_tracking_server_user_name}:{mlflow_tracking_server_password}@{mlflow_tracking_server_host}",
    )
    mlflow.set_experiment(mlflow_experiment_name)

    df = pd.DataFrame(flight_data)

    df["total_flight_hours"] = df["flight_duration_hours"] * df["number_of_flights"]
    df = df[df["total_flight_hours"] > 500]

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / Path(f"flight_data_{experiment_number}.csv")
        df.to_csv(file_path, index=False)
        with mlflow.start_run():
            mlflow.log_artifact(str(file_path))
            return df.head()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    ray.init()
    logger.info(ray.cluster_resources())

    mlflow_tracking_server_host = config.MLFLOW_TRACKING_SERVER_HOST
    mlflow_tracking_server_user_name = config.MLFLOW_TRACKING_USERNAME
    mlflow_tracking_server_password = config.MLFLOW_TRACKING_PASSWORD
    mlflow_experiment_name = config.MLFLOW_EXPERIMENT_NAME

    flight_data_list = [
        {
            "aircraft_type": ["A320", "B737", "A380", "B787", "A350"],
            "flight_duration_hours": [2, 3, 12, 14, 8],
            "number_of_flights": [300, 250, 50, 75, 100],
        },
        {
            "aircraft_type": ["B747", "A330", "B777", "A340", "B757"],
            "flight_duration_hours": [13, 6, 15, 12, 5],
            "number_of_flights": [60, 120, 80, 90, 150],
        },
        {
            "aircraft_type": ["A321", "B767", "A310", "B727", "A318"],
            "flight_duration_hours": [5, 9, 7, 4, 3],
            "number_of_flights": [180, 70, 130, 200, 220],
        },
        {
            "aircraft_type": ["B787-9", "A350-1000", "A330-900", "B777-300", "A321neo"],
            "flight_duration_hours": [10, 11, 9, 13, 6],
            "number_of_flights": [90, 80, 70, 60, 150],
        },
        {
            "aircraft_type": ["A380", "B777", "A350", "B787", "A330"],
            "flight_duration_hours": [12, 14, 10, 8, 6],
            "number_of_flights": [100, 90, 110, 130, 120],
        },
    ]

    tasks = [
        process_flight_data.remote(
            mlflow_tracking_server_host,
            mlflow_tracking_server_user_name,
            mlflow_tracking_server_password,
            mlflow_experiment_name,
            flight_data,
            i,
        )
        for i, flight_data in enumerate(flight_data_list)
    ]

    results = ray.get(tasks)
    for i, df_head in enumerate(results):
        logger.info(f"Experiment {i}")
        logger.info(f"{df_head = }")

    ray.shutdown()
