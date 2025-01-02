import logging
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

logger = logging.getLogger(__name__)


def main(prompt: str, output_path: Path, num_inference_steps) -> None:
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Use float16 for better memory efficiency
        safety_checker=None,
    )

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    try:
        image = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5,  # Controls how much the image generation follows the prompt
        ).images[0]

        image.save(output_path)
        logger.info(f"Image successfully generated and saved to {output_path}")

    except Exception as e:
        logger.exception(f"Error generating image: {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    prompt = "A beautiful sunset over mountains"
    output_path = Path("image.png")

    main(prompt=prompt, output_path=output_path, num_inference_steps=50)
