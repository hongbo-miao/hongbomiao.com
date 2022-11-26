import pandas as pd
from prefect import task


@task
def get_top_routes(trips: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    routes = (
        trips.value_counts(subset=["pulocationid", "dolocationid"], ascending=False)
        .to_frame("count")
        .reset_index()
        .merge(zones, left_on="pulocationid", right_on="locationid", how="inner")
        .rename(columns={"zone": "pulocation_zone", "borough": "pulocation_borough"})
        .drop(columns=["locationid", "shape_area", "shape_leng", "the_geom"])
        .merge(zones, left_on="dolocationid", right_on="locationid", how="inner")
        .rename(columns={"zone": "dolocation_zone", "borough": "dolocation_borough"})
        .drop(columns=["locationid", "shape_area", "shape_leng", "the_geom"])
    )
    return routes
