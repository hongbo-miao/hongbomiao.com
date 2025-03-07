import daft
import ray
from prefect import flow, get_run_logger


@flow
def hm_daft_analysis() -> None:
    logger = get_run_logger()

    ray_address = "ray://hm-ray-cluster-head-svc.production-hm-ray-cluster:10001"
    ray.init(
        ray_address,
        runtime_env={
            "pip": [
                "getdaft==0.4.6",
            ],
        },
    )
    daft.context.set_runner_ray(ray_address)

    df = daft.from_pydict(
        {
            "a": [3, 2, 5, 6, 1, 4],
            "b": [True, False, False, True, True, False],
        },
    )
    df = df.where(df["b"]).sort(df["a"])
    df = df.collect()
    logger.info(f"{df = }")


if __name__ == "__main__":
    hm_daft_analysis()
