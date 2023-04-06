from prefect.infrastructure import KubernetesJob

job = KubernetesJob(
    image="ghcr.io/hongbo-miao/hm-prefect-upload-to-influxdb:latest",
    namespace="hm-prefect",
    image_pull_policy="Always",
)
job.save("upload-to-influxdb-kubernetes-job-block", overwrite=True)
