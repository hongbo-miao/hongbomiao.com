from feast import FeatureStore

store = FeatureStore(repo_path="./driver_features")
feature_service = store.get_feature_service("driver_activity")
features = store.get_online_features(
    features=feature_service,
    entity_rows=[
        # {join_key: entity_value}
        {"driver_id": 1004},
        {"driver_id": 1005},
    ],
).to_dict()
