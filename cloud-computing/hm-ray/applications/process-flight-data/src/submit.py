from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient("https://ray.hongbomiao.com")
client.submit_job(
    entrypoint="python src/main.py",
    runtime_env={
        "working_dir": "./",
        "pip": [
            "mlflow==2.14.1",
            "polars==1.25.2",
        ],
    },
)
