from prefect.infrastructure import KubernetesJob

job = KubernetesJob(
    image="ghcr.io/hongbo-miao/hm-prefect-collect-data:latest",
    namespace="hm-prefect",
    image_pull_policy="Always",
)
job.save("collect-data-kubernetes-job-block", overwrite=True)
