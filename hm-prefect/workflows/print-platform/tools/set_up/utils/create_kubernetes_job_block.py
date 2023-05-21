from prefect.infrastructure import KubernetesJob


async def create_kubernetes_job_block(flow_name: str) -> None:
    await KubernetesJob(
        image=f"ghcr.io/hongbo-miao/hm-prefect-{flow_name}:latest",
        namespace="hm-prefect",
        image_pull_policy="Always",
    ).save(f"{flow_name}-kubernetes-job-block", overwrite=True)
