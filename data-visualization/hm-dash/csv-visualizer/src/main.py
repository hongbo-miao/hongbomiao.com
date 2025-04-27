import plotly.express as px
import polars as pl
from dash import Dash, Input, Output, callback, dcc, html
from plotly.graph_objs import Figure


def main() -> None:
    df = pl.read_csv(
        "https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv",
    )
    countries = df.select("country").unique().to_series().to_list()

    app = Dash()
    app.layout = html.Div(
        [
            html.H1(children="Dash App", style={"textAlign": "center"}),
            dcc.Dropdown(
                options=countries,
                value="United States",
                id="dropdown-selection",
            ),
            dcc.Graph(id="graph-content"),
        ],
    )

    @callback(
        Output("graph-content", "figure"),
        Input("dropdown-selection", "value"),
    )
    def update_graph(value: str) -> Figure:
        new_df = df.filter(pl.col("country") == value)
        return px.line(new_df, x="year", y="pop")

    app.run(debug=True)


if __name__ == "__main__":
    main()
