import logging

import pandas as pd
from pandas import option_context

logger = logging.getLogger(__name__)

with option_context(
    "display.max_rows",
    10,
    "display.max_columns",
    None,
    "display.width",
    500,
):
    df = pd.read_parquet("driver_features/data/driver_stats.parquet")
    logger.info(df)
