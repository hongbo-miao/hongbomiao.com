from typing import Any

import pandas as pd


def write_params(params: dict[str, Any]) -> None:
    df = pd.DataFrame.from_dict(params, orient="index", columns=["Value"])
    with open("output/reports/params.txt", "w") as f:
        f.write(df.to_markdown())
