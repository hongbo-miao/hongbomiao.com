import boto3
from tritonclient.http import InferenceServerClient, InferResult


def predict_by_tensorrt_model(
    sagemaker_runtime_client: boto3.client,
    endpoint_name: str,
    request_body: bytes,
    header_length: int,
) -> InferResult:
    # "Inference-Header-Content-Length" is not allowed as custom headers are not allowed in SageMaker
    header_name = "application/vnd.sagemaker-triton.binary+json;json-header-size"
    res = sagemaker_runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType=f"{header_name}={header_length}",
        Body=request_body,
        TargetModel="resnet_trt_v0.tar.gz",
    )
    return InferenceServerClient.parse_response_body(
        res["Body"].read(),
        header_length=int(res["ContentType"][len(f"{header_name}=") :]),
    )
