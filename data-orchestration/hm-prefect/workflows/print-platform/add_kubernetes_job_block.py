from prefect.infrastructure import KubernetesJob

job = KubernetesJob(
    image="ghcr.io/hongbo-miao/hm-prefect-print-platform:latest",
    namespace="hm-prefect",
    image_pull_policy="Always",
)
job.save("print-platform-kubernetes-job-block", overwrite=True)
