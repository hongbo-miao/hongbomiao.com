from kfp import client, dsl


@dsl.component
def add(a: int, b: int) -> int:
    return a + b


@dsl.component
def multiply(a: int, b: int) -> int:
    return a * b


@dsl.pipeline
def calculate(a: int, b: int):
    add_task = add(a=a, b=b)
    multiply(a=add_task.output, b=10)


if __name__ == "__main__":
    kfp_client = client.Client(host="https://kubeflow.hongbomiao.com")
    run = kfp_client.create_run_from_pipeline_func(
        calculate,
        arguments={"a": 1, "b": 2},
    )
