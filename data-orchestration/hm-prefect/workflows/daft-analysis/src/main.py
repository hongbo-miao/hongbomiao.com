import daft
import ray
from daft import DataCatalogType
from daft.io.catalog import DataCatalogTable
from prefect import flow, get_run_logger


@flow
def hm_daft_analysis() -> None:
    logger = get_run_logger()

    ray_address = "ray://hm-ray-cluster-head-svc.production-hm-ray-cluster.svc:10001"
    ray.init(
        ray_address,
        runtime_env={
            "pip": [
                "getdaft[aws,deltalake,ray]==0.4.6",
            ],
        },
    )
    daft.context.set_runner_ray(ray_address)

    table = DataCatalogTable(
        catalog=DataCatalogType.GLUE,
        database_name="production_motor_db",
        table_name="motor_data",
    )

    df = daft.read_deltalake(table)
    df = df.where(df["_event_id"] == "ad7953cd-6d49-4929-8180-99555bebc255")
    df = df.collect()
    logger.info(f"{df = }")


if __name__ == "__main__":
    hm_daft_analysis()
