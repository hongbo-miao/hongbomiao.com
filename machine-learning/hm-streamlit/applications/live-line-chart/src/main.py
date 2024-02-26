import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.DataFrame(columns=["value"])


def main() -> None:
    st.title("Live Line Chart")
    generator = np.random.default_rng(42)
    max_data_points = 100
    prev_time = None
    df = get_data()
    placeholder = st.empty()
    while True:
        current_time = datetime.now()
        new_data_point = (
            generator.standard_normal(1) / 10.0 + df.loc[prev_time]
            if prev_time
            else generator.standard_normal(1)
        )
        df.loc[current_time] = new_data_point
        prev_time = current_time

        # Remove old timestamps if the DataFrame exceeds the maximum size
        if len(df) > max_data_points:
            df = df.iloc[1:]

        with placeholder.container():
            st.line_chart(df, height=200)
            time.sleep(0.01)


if __name__ == "__main__":
    main()
