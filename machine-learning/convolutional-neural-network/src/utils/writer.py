from pathlib import Path
from typing import Any

import pandas as pd


def write_params(params: dict[str, Any]) -> None:
    df = pd.DataFrame.from_dict(params, orient="index", columns=["Value"])
    with Path("output/reports/params.txt").open("w") as f:
        f.write(df.to_markdown())
