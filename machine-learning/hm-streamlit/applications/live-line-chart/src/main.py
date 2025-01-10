import time
from datetime import UTC, datetime

import numpy as np
import polars as pl
import streamlit as st


@st.cache_data
def get_data() -> pl.DataFrame:
    return pl.DataFrame(
        {"timestamp": [], "value1": [], "value2": []},
        schema={"timestamp": pl.Datetime, "value1": pl.Float64, "value2": pl.Float64},
    )


def main() -> None:
    st.title("Live Line Chart")
    generator = np.random.default_rng(42)
    max_data_points = 100
    prev_values = None
    df = get_data()
    placeholder = st.empty()

    while True:
        current_time = datetime.now(tz=UTC)
        new_data_point = (
            generator.standard_normal(2) / 10.0 + prev_values
            if prev_values is not None
            else generator.standard_normal(2)
        )
        prev_values = new_data_point

        # Add new row to DataFrame
        df = pl.concat(
            [
                df,
                pl.DataFrame(
                    {
                        "timestamp": [current_time],
                        "value1": [new_data_point[0]],
                        "value2": [new_data_point[1]],
                    },
                ),
            ],
        )

        # Remove old timestamps if the DataFrame exceeds the maximum size
        if len(df) > max_data_points:
            df = df.slice(1, len(df))

        with placeholder.container():
            st.header("Chart")
            st.line_chart(df.select(["value1", "value2"]), height=200)

            st.header("Table")
            st.dataframe(df)

            time.sleep(0.01)


if __name__ == "__main__":
    main()
