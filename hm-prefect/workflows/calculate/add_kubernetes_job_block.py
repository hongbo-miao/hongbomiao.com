from prefect.infrastructure import KubernetesJob

job = KubernetesJob(
    image="ghcr.io/hongbo-miao/hm-prefect-calculate:latest",
    namespace="hm-prefect",
    image_pull_policy="Always",
)
job.save("calculate-kubernetes-job-block", overwrite=True)
