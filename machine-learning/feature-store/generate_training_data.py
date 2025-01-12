import logging
from datetime import UTC, datetime, timedelta

import pandas as pd
from feast import FeatureStore
from pandas import option_context

logger = logging.getLogger(__name__)

# The entity dataframe is the dataframe we want to enrich with feature values
entity_df = pd.DataFrame.from_dict(
    {
        "driver_id": [1001, 1002, 1003],
        "label_driver_reported_satisfaction": [1, 5, 3],
        "event_timestamp": [
            datetime.now(tz=UTC) - timedelta(minutes=11),
            datetime.now(tz=UTC) - timedelta(minutes=36),
            datetime.now(tz=UTC) - timedelta(minutes=73),
        ],
    },
)

store = FeatureStore(repo_path="./driver_features")

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "driver_hourly_stats:conv_rate",
        "driver_hourly_stats:acc_rate",
        "driver_hourly_stats:avg_daily_trips",
    ],
).to_df()

with option_context(
    "display.max_rows",
    10,
    "display.max_columns",
    None,
    "display.width",
    500,
):
    logger.info("----- Feature schema -----")
    logger.info(training_df.info())
    logger.info("----- Example features -----")
    logger.info(training_df.head())
