import logging

import torch
from PIL import Image
from shared.image.utils.create_altitude_image import create_altitude_image
from vision_language_action_lib.language.utils.load_language_model import (
    load_language_model,
)
from vision_language_action_lib.policy.models.flow_matching_policy import (
    FlowMatchingPolicy,
)
from vision_language_action_lib.vision.models.vision_projection import (
    create_vision_projection,
)
from vision_language_action_lib.vision.utils.encode_image import encode_image
from vision_language_action_lib.vision.utils.load_vision_model import load_vision_model

logger = logging.getLogger(__name__)

DINOV3_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
QWEN3_EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
CHECKPOINT_DIRECTORY = "output/checkpoints"


def validate_action_predictions() -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )
    logger.info(f"Using device: {device}")

    logger.info("Loading vision model...")
    vision_model, _ = load_vision_model(DINOV3_MODEL_ID, device)
    vision_dimension = vision_model.config.hidden_size

    logger.info("Loading embedding model...")
    embedding_model, tokenizer, _ = load_language_model(
        QWEN3_EMBEDDING_MODEL_ID,
        device,
    )
    embedding_dimension = embedding_model.config.hidden_size

    logger.info("Loading vision projection checkpoint...")
    vision_projection = create_vision_projection(
        vision_dimension,
        embedding_dimension,
        device,
    )
    vision_projection.load_state_dict(
        torch.load(f"{CHECKPOINT_DIRECTORY}/vision_projection.pt", map_location=device),
    )
    vision_projection.eval()

    logger.info("Loading policy checkpoint...")
    context_dimension = embedding_dimension * 2
    policy = FlowMatchingPolicy(
        context_dimension=context_dimension,
        action_dimension=6,
        hidden_dimension=2048,
        layer_count=8,
    ).to(device)
    policy.load_state_dict(
        torch.load(
            f"{CHECKPOINT_DIRECTORY}/flow_matching_policy.pt",
            map_location=device,
        ),
    )
    policy.eval()

    def get_action(image: Image.Image, instruction: str, seed: int = 42) -> list[float]:
        vision_features = encode_image(image, vision_model, device)
        vision_tokens = vision_projection(vision_features)
        vision_pooled = vision_tokens.mean(dim=1)

        tokens = tokenizer(
            instruction,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        with torch.no_grad():
            outputs = embedding_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            text_embedding = outputs.last_hidden_state.mean(dim=1)

        context = torch.cat([vision_pooled, text_embedding], dim=-1).float()

        torch.manual_seed(seed)
        action = torch.randn(1, 6, device=device, dtype=torch.float32)
        delta_time = 0.1

        for step in range(10):
            t = torch.full((1, 1), step / 10, device=device, dtype=torch.float32)
            velocity = policy(context, action, t)
            action = action + velocity * delta_time

        action = torch.clamp(action, -2, 2)
        return action.squeeze().cpu().tolist()

    high_altitude_image = create_altitude_image(simulated_altitude=7.0)
    low_altitude_image = create_altitude_image(simulated_altitude=1.0)

    logger.info("\nComparing delta_z for different instructions:\n")
    logger.info(
        f"{'Instruction':<20} {'Altitude':>10} {'delta_z':>10} {'Direction':>10}",
    )
    logger.info("-" * 54)

    landing_instructions = ["land", "landing", "descend", "go down"]
    takeoff_instructions = ["fly up", "take off", "ascend", "go up"]

    for instruction in landing_instructions:
        action = get_action(high_altitude_image, instruction)
        delta_z = action[2]
        direction = "UP" if delta_z > 0 else "DOWN"
        logger.info(f"{instruction:<20} {'HIGH':>10} {delta_z:>+10.4f} {direction:>10}")

    for instruction in takeoff_instructions:
        action = get_action(low_altitude_image, instruction)
        delta_z = action[2]
        direction = "UP" if delta_z > 0 else "DOWN"
        logger.info(f"{instruction:<20} {'LOW':>10} {delta_z:>+10.4f} {direction:>10}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    validate_action_predictions()
