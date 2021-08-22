import pandas as pd


def write_args(args):
    df = pd.DataFrame.from_dict(vars(args), orient="index", columns=["Value"])
    with open("reports/args.txt", "w") as outfile:
        outfile.write(df.to_markdown())
