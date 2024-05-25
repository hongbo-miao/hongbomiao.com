import logging
from pathlib import Path

import boto3
from utils.get_sample_image_binary import get_sample_image_binary
from utils.predict_by_pytorch_model import predict_by_pytorch_model
from utils.predict_by_tensorrt_model import predict_by_tensorrt_model


def main() -> None:
    image_path = Path("data/dog.jpg")
    sagemaker_client = boto3.client("sagemaker-runtime")
    endpoint_name = "resnet50-mme-gpu-ep-2024-05-25-05-40-24"

    # TensorRT
    input_name = "input"
    output_name = "output"
    request_body, header_length = get_sample_image_binary(
        image_path, input_name, output_name
    )
    result = predict_by_tensorrt_model(
        sagemaker_client, endpoint_name, request_body, header_length
    )
    logging.info(result.as_numpy(output_name))

    # PyTorch
    input_name = "INPUT__0"
    output_name = "OUTPUT__0"
    request_body, header_length = get_sample_image_binary(
        image_path, input_name, output_name
    )
    result = predict_by_pytorch_model(
        sagemaker_client, endpoint_name, request_body, header_length
    )
    print(type(result))
    logging.info(result.as_numpy(output_name))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
