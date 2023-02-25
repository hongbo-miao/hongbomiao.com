import pandas as pd
import pytest
from src.utils.df import union_all


class TestGroup:
    @pytest.fixture
    def expected_df(self):
        return pd.DataFrame(
            [
                ["dog_1", "cat_1"],
                ["dog_2", "cat_2"],
                ["dog_3", "cat_3"],
            ],
            columns=["dog", "cat"],
        )

    def test_union_dataframes(self, expected_df):
        df1 = pd.DataFrame(
            [
                ["dog_1", "cat_1"],
                ["dog_2", "cat_2"],
            ],
            columns=["dog", "cat"],
        )
        df2 = pd.DataFrame(
            [
                ["dog_3", "cat_3"],
            ],
            columns=["dog", "cat"],
        )
        pd.testing.assert_frame_equal(union_all(df1, df2), expected_df)

    def test_union_one_dataframe(self, expected_df):
        df1 = pd.DataFrame(
            [
                ["dog_1", "cat_1"],
                ["dog_2", "cat_2"],
                ["dog_3", "cat_3"],
            ],
            columns=["dog", "cat"],
        )
        pd.testing.assert_frame_equal(union_all(df1), expected_df)

    def test_not_valid_end_time(self):
        with pytest.raises(ValueError, match="No objects to concatenate"):
            union_all()
