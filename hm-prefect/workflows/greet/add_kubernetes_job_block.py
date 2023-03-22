from prefect.infrastructure import KubernetesJob

job = KubernetesJob(
    image="ghcr.io/hongbo-miao/hm-prefect-greet:latest",
    namespace="hm-prefect",
    image_pull_policy="Always",
)
job.save("greet-kubernetes-job-block", overwrite=True)
