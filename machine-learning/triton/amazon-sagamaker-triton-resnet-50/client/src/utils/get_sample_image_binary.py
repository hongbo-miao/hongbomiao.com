from pathlib import Path

import numpy as np
import tritonclient.http as httpclient
from utils.get_sample_image import get_sample_image


def get_sample_image_binary(
    image_path: Path, input_name: str, output_name: str
) -> tuple[bytes, int]:
    inputs = []
    inputs.append(httpclient.InferInput(input_name, [1, 3, 224, 224], "FP32"))
    input_data = np.array(get_sample_image(image_path), dtype=np.float32)
    input_data = np.expand_dims(input_data, axis=0)
    inputs[0].set_data_from_numpy(input_data, binary_data=True)

    outputs = []
    outputs.append(httpclient.InferRequestedOutput(output_name, binary_data=True))

    request_body, header_length = (
        httpclient.InferenceServerClient.generate_request_body(inputs, outputs=outputs)
    )
    return request_body, header_length
