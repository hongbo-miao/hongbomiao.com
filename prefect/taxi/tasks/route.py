import pandas as pd

from prefect import get_run_logger, task


@task
def get_top_routes(trips: pd.DataFrame, top_count: int) -> pd.DataFrame:
    routes = (
        trips.value_counts(subset=["pulocationid", "dolocationid"], ascending=False)
        .to_frame("count")
        .reset_index()
    )
    mask = routes["pulocationid"] != routes["dolocationid"]
    routes = routes[mask].reset_index(drop=True)
    return routes[:top_count]


@task
def print_routes(routes: pd.DataFrame, zones: pd.DataFrame) -> None:
    logger = get_run_logger()
    for _, route in routes.iterrows():
        u = zones.loc[route["pulocationid"]]["zone"]
        v = zones.loc[route["dolocationid"]]["zone"]
        count = route["count"]
        logger.info(f"{u} -> {v}: {count}")
