import logging
from pathlib import Path

import geodatasets
import geopandas
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def main() -> None:
    gdf = geopandas.read_file(geodatasets.get_path("geoda.chicago_commpop"))
    logger.info(gdf.head())
    gdf.to_parquet(Path("data/chicago_commpop.parquet"))
    gdf.plot(
        column="POP2010",
        legend=True,
        scheme="quantiles",
        figsize=(15, 10),
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "red",
            "hatch": "///",
            "label": "Missing values",
        },
    )
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
