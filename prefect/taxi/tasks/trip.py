import io

import pandas as pd
from utils.enum import CalcMethod

from prefect import task


@task
def get_trips(data: bytes) -> pd.DataFrame:
    df = pd.read_parquet(io.BytesIO(data), engine="pyarrow")
    df = df.rename(str.lower, axis="columns")
    df["store_and_fwd_flag"] = df["store_and_fwd_flag"].astype("bool")
    mask = df["total_amount"] > 0
    return df[mask]


@task
def get_average_amount(trips: pd.DataFrame, calc_method: CalcMethod) -> float:
    match calc_method:
        case CalcMethod.MEAN:
            return trips["total_amount"].mean()
        case CalcMethod.MEDIAN:
            return trips["total_amount"].median()
        case _:
            raise ValueError(f"Unknown calc_method: {calc_method}")


@task
def get_average_trip_distance(trips: pd.DataFrame, calc_method: CalcMethod) -> float:
    match calc_method:
        case CalcMethod.MEAN:
            return trips["trip_distance"].mean()
        case CalcMethod.MEDIAN:
            return trips["trip_distance"].median()
        case _:
            raise ValueError(f"Unknown calc_method: {calc_method}")


@task
def get_price_per_mile(trip_distance: float, amount: float) -> float:
    return amount / trip_distance


@task
def get_trip(trip_id: int, trips: pd.DataFrame) -> pd.Series:
    return trips.loc[trip_id, :]


@task
def get_is_higher_than_unit_price(trip: pd.Series, average_amount: float) -> bool:
    return trip["total_amount"] / trip["trip_distance"] > average_amount
