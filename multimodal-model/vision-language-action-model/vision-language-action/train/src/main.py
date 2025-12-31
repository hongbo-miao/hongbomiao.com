import logging
from pathlib import Path

import torch
from shared.training.utils.train_flow_matching_policy import train_flow_matching_policy
from vision_language_action_lib.language.utils.load_language_model import (
    load_language_model,
)
from vision_language_action_lib.policy.models.flow_matching_policy import (
    FlowMatchingPolicy,
)
from vision_language_action_lib.vision.models.vision_projection import (
    create_vision_projection,
)
from vision_language_action_lib.vision.utils.load_vision_model import load_vision_model

logger = logging.getLogger(__name__)

CHECKPOINT_DIRECTORY = Path("output/checkpoints")

DINOV3_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
QWEN3_MODEL_ID = "Qwen/Qwen3-0.6B"

SAMPLE_COUNT = 5000
BATCH_SIZE = 64
EPOCH_COUNT = 500
LEARNING_RATE = 1e-4


def main() -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )
    logger.info(f"Using device: {device}")

    logger.info("Loading vision model...")
    vision_model, _ = load_vision_model(
        model_id=DINOV3_MODEL_ID,
        device=device,
    )
    vision_dimension = vision_model.config.hidden_size

    logger.info("Loading language model...")
    language_model, tokenizer, _ = load_language_model(
        model_id=QWEN3_MODEL_ID,
        device=device,
    )
    language_dimension = language_model.config.hidden_size

    logger.info("Creating vision projection...")
    vision_projection = create_vision_projection(
        vision_dimension=vision_dimension,
        language_dimension=language_dimension,
        device=device,
    )

    logger.info("Creating Flow Matching policy...")
    context_dimension = (
        language_dimension * 2
    )  # vision_pooled + text_embedding concatenated
    policy = FlowMatchingPolicy(
        context_dimension=context_dimension,
        action_dimension=6,
        hidden_dimension=2048,
        layer_count=8,
    ).to(device)

    logger.info("Starting training...")
    train_flow_matching_policy(
        policy=policy,
        vision_model=vision_model,
        language_model=language_model,
        tokenizer=tokenizer,
        vision_projection=vision_projection,
        device=device,
        sample_count=SAMPLE_COUNT,
        batch_size=BATCH_SIZE,
        epoch_count=EPOCH_COUNT,
        learning_rate=LEARNING_RATE,
        checkpoint_directory=CHECKPOINT_DIRECTORY,
    )

    logger.info("Training complete!")
    logger.info(f"Checkpoint saved to {CHECKPOINT_DIRECTORY}/flow_matching_policy.pt")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
