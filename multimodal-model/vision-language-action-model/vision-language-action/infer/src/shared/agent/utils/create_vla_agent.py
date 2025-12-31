import logging
from pathlib import Path

import torch
from shared.agent.services.vla_agent import VLAAgent
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


def create_vla_agent(
    vision_model_id: str,
    language_model_id: str,
    flow_matching_policy_checkpoint_path: Path,
    vision_projection_checkpoint_path: Path,
    flow_matching_policy_action_dimension: int,
    flow_matching_policy_hidden_dimension: int,
    flow_matching_policy_layer_count: int,
    device: torch.device,
) -> VLAAgent:
    logger.info("Creating VLA Agent")
    logger.info(f"Device: {device}")

    logger.info("Loading vision model (DINOv3)")
    vision_model, _ = load_vision_model(
        model_id=vision_model_id,
        device=device,
    )

    vision_dimension = vision_model.config.hidden_size
    logger.info(f"Vision dimension: {vision_dimension}")

    logger.info("Loading language model (Qwen3)")
    language_model, tokenizer, _ = load_language_model(
        model_id=language_model_id,
        device=device,
    )

    language_dimension = language_model.config.hidden_size
    logger.info(f"Language dimension: {language_dimension}")

    logger.info("Creating vision projection layer")
    vision_projection = create_vision_projection(
        vision_dimension=vision_dimension,
        language_dimension=language_dimension,
        device=device,
    )

    if not vision_projection_checkpoint_path.exists():
        msg = f"Vision projection checkpoint not found: {vision_projection_checkpoint_path}"
        raise FileNotFoundError(
            msg,
        )
    logger.info(
        f"Loading vision projection checkpoint from {vision_projection_checkpoint_path}",
    )
    vision_projection.load_state_dict(
        torch.load(vision_projection_checkpoint_path, map_location=device),
    )
    vision_projection.eval()
    logger.info("Vision projection checkpoint loaded successfully")

    logger.info("Creating Flow Matching policy")
    # vision_pooled + text_embedding concatenated
    context_dimension = language_dimension * 2
    flow_matching_policy = FlowMatchingPolicy(
        context_dimension=context_dimension,
        action_dimension=flow_matching_policy_action_dimension,
        hidden_dimension=flow_matching_policy_hidden_dimension,
        layer_count=flow_matching_policy_layer_count,
    ).to(device)

    if not flow_matching_policy_checkpoint_path.exists():
        msg = f"Flow matching policy checkpoint not found: {flow_matching_policy_checkpoint_path}"
        raise FileNotFoundError(
            msg,
        )
    logger.info(
        f"Loading flow matching policy checkpoint from {flow_matching_policy_checkpoint_path}",
    )
    flow_matching_policy.load_state_dict(
        torch.load(flow_matching_policy_checkpoint_path, map_location=device),
    )
    flow_matching_policy.eval()
    logger.info("Flow matching policy checkpoint loaded successfully")

    agent = VLAAgent(
        vision_model=vision_model,
        language_model=language_model,
        tokenizer=tokenizer,
        vision_projection=vision_projection,
        policy=flow_matching_policy,
        device=device,
    )

    logger.info("VLA Agent created successfully")

    return agent
