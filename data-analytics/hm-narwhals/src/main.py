import logging

import polars as pl
from utils.standard_scaler import StandardScaler

logger = logging.getLogger(__name__)


def main() -> None:
    df_train = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 7]})
    df_test = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 7]})
    scaler = StandardScaler()
    scaler.fit(df_train)
    logger.info(scaler.transform(df_test).collect())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
