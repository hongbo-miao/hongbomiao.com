import numpy as np
import pandas as pd
import streamlit as st


def main() -> None:
    generator = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "col1": generator.standard_normal(1000) / 50 + 37.76,
            "col2": generator.standard_normal(1000) / 50 + -122.4,
            "col3": generator.standard_normal(1000) * 100,
            "col4": generator.standard_normal((1000, 4)).tolist(),
        }
    )
    st.map(df, latitude="col1", longitude="col2", size="col3", color="col4")


if __name__ == "__main__":
    main()
