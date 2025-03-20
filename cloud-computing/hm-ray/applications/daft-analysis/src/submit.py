from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient("https://ray.hongbomiao.com")
client.submit_job(
    entrypoint="python src/main.py",
    runtime_env={
        "working_dir": "./",
        "pip": [
            "daft[aws,deltalake,sql]==0.4.7",
            "trino[sqlalchemy]==0.333.0",
        ],
    },
)
