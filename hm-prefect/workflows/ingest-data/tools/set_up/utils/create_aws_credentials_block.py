from prefect_aws import AwsCredentials


async def create_aws_credentials_block(
    flow_name: str,
    aws_default_region: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
) -> None:
    await AwsCredentials(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_default_region,
    ).save(f"{flow_name}-aws-credentials-block", overwrite=True)
