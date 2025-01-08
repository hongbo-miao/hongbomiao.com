import logging
import secrets
from datetime import datetime
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

logger = logging.getLogger(__name__)


class StableDiffusionGenerator:
    @staticmethod
    def get_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def create_pipeline(
        model_id: str,
        is_safety_checker_enabled: bool = True,
    ) -> StableDiffusionPipeline:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker",
                torch_dtype=torch.float16,
            )
            if is_safety_checker_enabled
            else None,
        )

        # Use DPM++ 2M Karras scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )

        device = StableDiffusionGenerator.get_device()
        pipe = pipe.to(device)

        # Enable attention slicing to reduce memory usage by processing attention in chunks
        pipe.enable_attention_slicing()
        return pipe

    @staticmethod
    def generate_images(
        pipe: StableDiffusionPipeline,
        prompt: str,
        output_dir: Path,
        negative_prompt: str | None = None,
        image_number: int = 1,
        width: int = 768,
        height: int = 768,
        inference_step_number: int = 50,
        guidance_scale: float = 7.5,
        seed: int | None = None,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set up seed for reproducibility
        if seed is None:
            seed = secrets.randbelow(2**32)

        device = StableDiffusionGenerator.get_device()
        generator = torch.Generator(device).manual_seed(seed)
        try:
            # Generate images
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=image_number,
                width=width,
                height=height,
                inference_step_number=inference_step_number,
                guidance_scale=guidance_scale,
                generator=generator,
            )

            # Save images
            for idx, image in enumerate(result.images):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = output_dir / f"{timestamp}_{idx}.png"
                image.save(path)

        except Exception:
            logger.exception("Error generating images")


def main() -> None:
    pipe = StableDiffusionGenerator.create_pipeline(
        model_id="stabilityai/stable-diffusion-2-1",
        is_safety_checker_enabled=True,
    )

    StableDiffusionGenerator.generate_images(
        pipe=pipe,
        prompt="A beautiful sunset over mountains, highly detailed, majestic",
        negative_prompt="blur, low quality, bad anatomy, worst quality, low resolution, watermark, text, signature, copyright, logo, brand name",
        image_number=1,
        # Stable Diffusion 2 default is 768x768
        width=768,
        height=768,
        inference_step_number=50,
        # Controls how much the image generation follows the prompt. Higher values = more prompt adherence
        guidance_scale=7.5,
        seed=None,
        output_dir=Path("output"),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
