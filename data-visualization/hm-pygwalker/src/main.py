import polars as pl
import pygwalker as pyg


def main() -> None:
    df = pl.read_csv(
        "https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv",
    )
    pyg.walk(
        dataset=df,
        default_tab="data",
        show_cloud_tool=False,
    )


if __name__ == "__main__":
    main()
