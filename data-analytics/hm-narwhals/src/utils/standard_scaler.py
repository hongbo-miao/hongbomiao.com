from typing import Self

import narwhals as nw
from narwhals.typing import DataFrameT, FrameT


class StandardScaler:
    @nw.narwhalify(eager_only=True)
    def fit(self: Self, df: DataFrameT) -> Self:
        self._means = {col: df[col].mean() for col in df.columns}
        self._standard_deviations = {col: df[col].std() for col in df.columns}
        self._columns = df.columns
        return self

    @nw.narwhalify
    def transform(self: Self, df: FrameT) -> FrameT:
        return df.with_columns(
            (nw.col(col) - self._means[col]) / self._standard_deviations[col]
            for col in self._columns
        )
