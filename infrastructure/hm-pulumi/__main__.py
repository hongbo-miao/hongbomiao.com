import pulumi
from pulumi_aws import s3

bucket = s3.Bucket("hm-production-bucket")

pulumi.export("bucket_name", bucket.id)
