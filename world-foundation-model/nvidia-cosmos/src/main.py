import logging
from pathlib import Path

from cosmos_oss.init import cleanup_environment, init_environment
from cosmos_predict2.config import InferenceArguments, InferenceType, SetupArguments
from cosmos_predict2.inference import Inference

logger = logging.getLogger(__name__)

# Cosmos-Predict2.5-2B
MODEL_NAME = "2B/post-trained"


def generate_video(
    inference: Inference,
    inference_arguments: InferenceArguments,
    output_directory: Path,
) -> list[str]:
    return inference.generate([inference_arguments], output_directory)


def main() -> None:
    output_directory = Path("output")

    setup_arguments = SetupArguments(
        output_dir=output_directory,
        model=MODEL_NAME,
        offload_diffusion_model=True,
        offload_text_encoder=True,
        offload_tokenizer=True,
        offload_guardrail_models=True,
    )

    logger.info(f"Loading model: {MODEL_NAME}")

    inference = Inference(setup_arguments)

    prompt = "A sleek, humanoid robot stands in a vast warehouse filled with neatly stacked cardboard boxes on industrial shelves. The robot's metallic body gleams under the bright, even lighting, highlighting its futuristic design and intricate joints."

    inference_arguments = InferenceArguments(
        name="video",
        prompt=prompt,
        inference_type=InferenceType.TEXT2WORLD,
        num_output_frames=30,
        resolution="256,448",
    )

    output_paths = generate_video(inference, inference_arguments, output_directory)
    for output_path in output_paths:
        logger.info(f"Video saved to: {output_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Sets TOKENIZERS_PARALLELISM=false to avoid HuggingFace tokenizer warnings, and sets up distributed training if RANK env var is set
    init_environment()

    main()

    # Destroys distributed process groups if running in distributed mode
    cleanup_environment()
