import logging
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

logger = logging.getLogger(__name__)

MODEL_ID = "openvla/openvla-7b"


def predict_action(
    model: AutoModelForVision2Seq,
    processor: AutoProcessor,
    image: Image.Image,
    instruction: str,
) -> list[float]:
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)

    action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    return action.tolist()


def main() -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )
    logger.info(f"Loading model: {MODEL_ID}")
    logger.info(f"Using device: {device}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    image_path = Path("data/image.jpg")
    image = Image.open(image_path).convert("RGB")
    instruction = "pick up the object"

    action = predict_action(model, processor, image, instruction)
    # Action format: [x, y, z, roll, pitch, yaw, gripper]
    # Example: [-0.00311655245808996, 0.00022339712872219974, 0.008073165028849059, 0.008801438395883376, 0.0005591315679690234, -0.011349297856583396, 0.1098039215686275]
    logger.info(f"Predicted action: {action}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
