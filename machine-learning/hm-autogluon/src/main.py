import logging

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor


def main(dataset_url: str, label: str) -> None:
    train_data = TabularDataset(dataset_url)
    predictor = TabularPredictor(label=label).fit(train_data)

    test_data = TabularDataset(dataset_url)

    y_pred = predictor.predict(test_data.drop(columns=[label]))
    logging.info(f"{y_pred = }")

    performance = predictor.evaluate(test_data)
    logging.info(f"{performance = }")

    leaderboard = predictor.leaderboard(test_data)
    logging.info(f"{type(leaderboard) = }")
    logging.info(f"{leaderboard = }")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    dataset_url = (
        "https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/train.csv"
    )
    label = "signature"
    main(dataset_url, label)
