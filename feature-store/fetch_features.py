from pprint import pprint

from feast import FeatureStore

store = FeatureStore(repo_path="./driver_features")

features = store.get_online_features(
    features=[
        "driver_hourly_stats:conv_rate",
        "driver_hourly_stats:acc_rate",
        "driver_hourly_stats:avg_daily_trips",
    ],
    entity_rows=[
        # {join_key: entity_value}
        {"driver_id": 1004},
        {"driver_id": 1005},
    ],
).to_dict()

pprint(features)
