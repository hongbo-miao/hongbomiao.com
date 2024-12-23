from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient("https://ray.internal.hongbomiao.com")
client.submit_job(
    entrypoint="python src/main.py",
    runtime_env={
        "working_dir": "./",
    },
)
