import logging

import torch
from PIL import Image
from shared.policy.utils.sample_action import sample_action
from transformers import PreTrainedModel, PreTrainedTokenizer
from vision_language_action_lib.policy.models.flow_matching_policy import (
    FlowMatchingPolicy,
)
from vision_language_action_lib.types.action_output import ActionOutput
from vision_language_action_lib.vision.models.vision_projection import VisionProjection
from vision_language_action_lib.vision.utils.encode_image import encode_image

logger = logging.getLogger(__name__)


class VLAAgent:
    """Vision-Language-Action Agent using DINOv3 + Qwen3 + Flow Matching."""

    def __init__(
        self,
        vision_model: PreTrainedModel,
        language_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        vision_projection: VisionProjection,
        policy: FlowMatchingPolicy,
        device: torch.device,
    ) -> None:
        self.vision_model = vision_model
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.vision_projection = vision_projection
        self.policy = policy
        self.device = device

        logger.info("VLA Agent initialized")

    def predict_action(
        self,
        image: Image.Image,
        instruction: str,
        num_integration_steps: int = 10,
    ) -> ActionOutput:
        """
        Predict action given image observation and language instruction.

        Args:
            image: Camera observation (PIL Image)
            instruction: Language instruction (e.g., "Land on the helipad")
            num_integration_steps: Number of ODE integration steps for Flow Matching

        Returns:
            ActionOutput with 6-DoF action deltas

        """
        logger.debug(f"Predicting action for instruction: {instruction}")

        vision_features = encode_image(
            image=image,
            model=self.vision_model,
            device=self.device,
        )

        vision_tokens = self.vision_projection(vision_features)

        tokens = self.tokenizer(
            instruction,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            text_embedding = outputs.last_hidden_state.mean(dim=1)

        vision_pooled = vision_tokens.mean(dim=1)
        pooled_context = torch.cat(
            [vision_pooled, text_embedding.to(vision_pooled.dtype)],
            dim=-1,
        )

        action = sample_action(
            policy=self.policy,
            context=pooled_context,
            num_integration_steps=num_integration_steps,
        )

        logger.info(
            f"Predicted action: dx={action.delta_x_mps:.4f} m/s, dy={action.delta_y_mps:.4f} m/s, dz={action.delta_z_mps:.4f} m/s, "
            f"droll={action.delta_roll_radps:.4f} rad/s, dpitch={action.delta_pitch_radps:.4f} rad/s, dyaw={action.delta_yaw_radps:.4f} rad/s",
        )

        return action
