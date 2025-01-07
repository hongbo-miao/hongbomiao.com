import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data
def get_data() -> pd.DataFrame:
    generator = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "latitude": generator.standard_normal(1000) / 50.0 + 37.76,
            "longitude": generator.standard_normal(1000) / 50.0 + -122.4,
            "size": generator.standard_normal(1000) * 100.0,
            "color": generator.standard_normal((1000, 4)).tolist(),
        },
    )


def main() -> None:
    st.title("Map")
    df = get_data()
    st.map(df, latitude="latitude", longitude="longitude", size="size", color="color")


if __name__ == "__main__":
    main()
