import boto3


def undeploy() -> None:
    model_name = "resnet-50"
    sagemaker_client = boto3.client(service_name="sagemaker")

    sagemaker_model_name = f"{model_name}-model"
    sagemaker_endpoint_config_name = f"{model_name}-endpoint-config"
    sagemaker_endpoint_name = f"{model_name}-endpoint"

    sagemaker_client.delete_model(ModelName=sagemaker_model_name)
    sagemaker_client.delete_endpoint_config(
        EndpointConfigName=sagemaker_endpoint_config_name,
    )
    sagemaker_client.delete_endpoint(EndpointName=sagemaker_endpoint_name)


if __name__ == "__main__":
    undeploy()
