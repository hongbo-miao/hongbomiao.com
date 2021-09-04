import pandas as pd


def write_params(params):
    df = pd.DataFrame.from_dict(params, orient="index", columns=["Value"])
    with open("output/reports/params.txt", "w") as f:
        f.write(df.to_markdown())
