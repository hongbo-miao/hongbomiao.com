from pathlib import Path

import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from utils.get_sample_image import get_sample_image


def get_sample_image_binary(
    image_path: Path,
    input_name: str,
    output_name: str,
) -> tuple[bytes, int]:
    input_data = np.expand_dims(
        np.array(get_sample_image(image_path), dtype=np.float32),
        axis=0,
    )
    input_tensor = InferInput(input_name, [1, 3, 224, 224], "FP32")
    input_tensor.set_data_from_numpy(input_data, binary_data=True)

    output_tensor = InferRequestedOutput(output_name, binary_data=True)

    request_body, header_length = InferenceServerClient.generate_request_body(
        [input_tensor],
        outputs=[output_tensor],
    )
    return request_body, header_length
