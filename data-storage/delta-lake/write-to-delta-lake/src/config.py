import os

from dotenv import load_dotenv

load_dotenv(".env.production.local")

parquet_path = os.getenv("PARQUET_PATH")
s3_path = os.getenv("S3_PATH")

aws_default_region = os.getenv("AWS_DEFAULT_REGION")
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
