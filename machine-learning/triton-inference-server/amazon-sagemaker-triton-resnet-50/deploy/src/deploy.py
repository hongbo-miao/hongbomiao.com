import logging
import time

import boto3
import sagemaker
from botocore.client import BaseClient

logger = logging.getLogger(__name__)


def check_endpoint_status(
    sagemaker_client: BaseClient,
    sagemaker_endpoint_name: str,
) -> None:
    while (
        status := sagemaker_client.describe_endpoint(
            EndpointName=sagemaker_endpoint_name,
        )["EndpointStatus"]
    ) == "Creating":
        print(f"Status: {status}")
        time.sleep(30)
    print(f"Status: {status}")


def deploy() -> None:
    model_name = "resnet-50"

    # `sagemaker_execution_role = sagemaker.get_execution_role()` only works in the Jupyter notebook hosted by Amazon SageMaker
    aws_account_id = boto3.client("sts").get_caller_identity()["Account"]
    sagemaker_execution_role = f"arn:aws:iam::{aws_account_id}:role/AmazonSageMakerExecutionRole-hm-sagemaker-notebook"

    sagemaker_session = sagemaker.Session(boto_session=boto3.Session())
    sagemaker_model_name = f"{model_name}-model"
    sagemaker_endpoint_config_name = f"{model_name}-endpoint-config"
    sagemaker_endpoint_name = f"{model_name}-endpoint"
    model_s3_url = f"s3://{sagemaker_session.default_bucket()}/{model_name}/"

    # Account mapping for SageMaker multi-model endpoints (MME) Triton image
    aws_account_id_dict = {
        "us-east-1": "785573368785",
        "us-east-2": "007439368137",
        "us-west-1": "710691900526",
        "us-west-2": "301217895009",
        "eu-west-1": "802834080501",
        "eu-west-2": "205493899709",
        "eu-west-3": "254080097072",
        "eu-north-1": "601324751636",
        "eu-south-1": "966458181534",
        "eu-central-1": "746233611703",
        "ap-east-1": "110948597952",
        "ap-south-1": "763008648453",
        "ap-northeast-1": "941853720454",
        "ap-northeast-2": "151534178276",
        "ap-southeast-1": "324986816169",
        "ap-southeast-2": "355873309152",
        "cn-northwest-1": "474822919863",
        "cn-north-1": "472730292857",
        "sa-east-1": "756306329178",
        "ca-central-1": "464438896020",
        "me-south-1": "836785723513",
        "af-south-1": "774647643957",
    }
    region = boto3.Session().region_name
    if region not in aws_account_id_dict.keys():
        raise ValueError("Unsupported region")
    base = "amazonaws.com.cn" if region.startswith("cn-") else "amazonaws.com"
    triton_server_image_uri = f"{aws_account_id_dict[region]}.dkr.ecr.{region}.{base}/sagemaker-tritonserver:22.07-py3"

    # Create a model
    sagemaker_client = boto3.client("sagemaker")
    res = sagemaker_client.create_model(
        ModelName=sagemaker_model_name,
        ExecutionRoleArn=sagemaker_execution_role,
        PrimaryContainer={
            "Image": triton_server_image_uri,
            "ModelDataUrl": model_s3_url,
            "Mode": "MultiModel",
        },
    )
    logger.info(f'Model Arn: {res["ModelArn"]}')

    # Create an endpoint config
    res = sagemaker_client.create_endpoint_config(
        EndpointConfigName=sagemaker_endpoint_config_name,
        ProductionVariants=[
            {
                "InstanceType": "ml.g4dn.4xlarge",
                "InitialVariantWeight": 1,
                "InitialInstanceCount": 1,
                "ModelName": sagemaker_model_name,
                "VariantName": "AllTraffic",
            },
        ],
    )
    logger.info(f'Endpoint Config Arn: {res["EndpointConfigArn"]}')

    # Create an endpoint
    res = sagemaker_client.create_endpoint(
        EndpointName=sagemaker_endpoint_name,
        EndpointConfigName=sagemaker_endpoint_config_name,
    )
    logger.info(f'Endpoint Arn: {res["EndpointArn"]}')

    check_endpoint_status(sagemaker_client, sagemaker_endpoint_name)

    # Perform auto-scaling of the endpoint based on GPU memory utilization
    # This is the format in which application autoscaling references the endpoint
    resource_id = "endpoint/" + sagemaker_endpoint_name + "/variant/" + "AllTraffic"
    auto_scaling_client = boto3.client("application-autoscaling")
    auto_scaling_client.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=1,
        MaxCapacity=5,
    )
    # GPUMemoryUtilization metric
    auto_scaling_client.put_scaling_policy(
        PolicyName="GPUUtil-ScalingPolicy",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",  # SageMaker supports only instance count
        PolicyType="TargetTrackingScaling",  # StepScaling, TargetTrackingScaling
        TargetTrackingScalingPolicyConfiguration={
            # Scale out when GPU utilization hits GPUUtilization target value.
            "TargetValue": 60.0,
            "CustomizedMetricSpecification": {
                "MetricName": "GPUUtilization",
                "Namespace": "/aws/sagemaker/Endpoints",
                "Dimensions": [
                    {"Name": "EndpointName", "Value": sagemaker_endpoint_name},
                    {"Name": "VariantName", "Value": "AllTraffic"},
                ],
                "Statistic": "Average",  # Average, Minimum, Maximum, SampleCount, Sum
                "Unit": "Percent",
            },
            "ScaleInCooldown": 600,
            "ScaleOutCooldown": 200,
        },
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    deploy()
