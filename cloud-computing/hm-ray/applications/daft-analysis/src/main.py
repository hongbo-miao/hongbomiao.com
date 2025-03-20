import daft
import ray
from daft import DataCatalogType
from daft.io.catalog import DataCatalogTable

TRINO_URL = "trino://trino_user@trino.hongbomiao.com:443/?source=trino-sqlalchemy"


def get_event_id(event_name: str) -> str:
    sql_query = f"""
        select cast(id as varchar) as id
        from postgresql.public.motor_events
        where name = '{event_name}'
    """  # noqa: S608
    event_df = daft.read_sql(sql_query, TRINO_URL)
    if event_df.count_rows() == 0:
        msg = f"No event found with name: {event_name}"
        raise ValueError(msg)
    return event_df.to_arrow()["id"][0].as_py()


@ray.remote(num_cpus=0.5, memory=4 * 10**9)  # 0.5-core CPU, 4 GB of memory
def analyze(event_name: str) -> None:
    event_id = get_event_id(event_name)

    database_name = "motor_db"
    table_name = "motor_data"
    column_names = [
        "_time",
        "current",
        "voltage",
        "temperature",
    ]
    table = DataCatalogTable(
        catalog=DataCatalogType.GLUE,
        database_name=database_name,
        table_name=table_name,
    )
    df = daft.read_deltalake(table)
    result_df = df.where(df["_event_id"] == event_id).select(*column_names)
    result_df.show(5)


if __name__ == "__main__":
    event_names = [
        "experiment-1",
        "experiment-2",
        "experiment-3",
    ]
    ray.init()
    analyze_tasks = [analyze.remote(event_name) for event_name in event_names]
    ray.get(analyze_tasks)
