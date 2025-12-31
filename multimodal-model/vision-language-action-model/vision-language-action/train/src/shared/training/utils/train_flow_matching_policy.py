import logging
from pathlib import Path

import torch
from shared.dataset.utils.generate_flight_demonstrations import (
    demonstrations_to_tensors,
    generate_flight_demonstrations,
)
from torch import Tensor
from torch.optim import AdamW
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from vision_language_action_lib.policy.models.flow_matching_policy import (
    FlowMatchingPolicy,
)
from vision_language_action_lib.vision.models.vision_projection import VisionProjection
from vision_language_action_lib.vision.utils.encode_image import encode_image

logger = logging.getLogger(__name__)


def train_flow_matching_policy(
    policy: FlowMatchingPolicy,
    vision_model: PreTrainedModel,
    language_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    vision_projection: VisionProjection,
    device: torch.device,
    sample_count: int = 500,
    batch_size: int = 16,
    epoch_count: int = 50,
    learning_rate: float = 1e-4,
    checkpoint_directory: Path | None = None,
) -> FlowMatchingPolicy:
    logger.info(f"Generating {sample_count} flight demonstrations...")
    demonstrations = generate_flight_demonstrations(sample_count=sample_count)
    images, instructions, action_tensor = demonstrations_to_tensors(
        demonstrations=demonstrations,
        device=device,
    )

    logger.info("Pre-computing vision features and text embeddings (frozen)...")
    vision_features_list, text_features_list = precompute_features(
        images=images,
        instructions=instructions,
        vision_model=vision_model,
        language_model=language_model,
        tokenizer=tokenizer,
        device=device,
    )
    logger.info(f"Pre-computed {len(vision_features_list)} samples")

    logger.info("Training vision projection + policy...")
    vision_projection.train()
    policy.train()

    trainable_parameters = list(vision_projection.parameters()) + list(
        policy.parameters(),
    )
    optimizer = AdamW(trainable_parameters, lr=learning_rate, weight_decay=0.01)

    from torch.optim.lr_scheduler import CosineAnnealingLR  # noqa: PLC0415

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epoch_count,
        eta_min=learning_rate * 0.1,
    )

    total_samples = len(vision_features_list)
    batch_count = (total_samples + batch_size - 1) // batch_size

    logger.info(f"Starting training: {epoch_count} epochs, batch size {batch_size}")

    best_loss = float("inf")

    for epoch in range(epoch_count):
        epoch_loss = 0.0
        indices = torch.randperm(total_samples).tolist()

        for batch_index in range(batch_count):
            start_index = batch_index * batch_size
            end_index = min(start_index + batch_size, total_samples)
            batch_indices = indices[start_index:end_index]

            batch_context = compute_context_batch(
                batch_indices=batch_indices,
                vision_features_list=vision_features_list,
                text_features_list=text_features_list,
                vision_projection=vision_projection,
            )
            batch_actions = action_tensor[batch_indices]

            optimizer.zero_grad()
            loss = policy.compute_loss(
                context=batch_context,
                target_action=batch_actions,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=0.5)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        average_loss = epoch_loss / batch_count

        if average_loss < best_loss:
            best_loss = average_loss
            if checkpoint_directory is not None:
                checkpoint_directory.mkdir(parents=True, exist_ok=True)
                policy_path = checkpoint_directory / "flow_matching_policy.pt"
                projection_path = checkpoint_directory / "vision_projection.pt"
                torch.save(policy.state_dict(), policy_path)
                torch.save(vision_projection.state_dict(), projection_path)
                logger.info(
                    f"Epoch {epoch + 1}: New best loss {average_loss:.6f}, saved to {checkpoint_directory}",
                )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch {epoch + 1}/{epoch_count}, Loss: {average_loss:.6f}, LR: {current_lr:.6f}",
            )

    logger.info(f"Training completed. Best loss: {best_loss:.6f}")

    vision_projection.eval()
    policy.eval()
    return policy


def precompute_features(
    images: list,
    instructions: list[str],
    vision_model: PreTrainedModel,
    language_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> tuple[list[Tensor], list[Tensor]]:
    vision_features_list = []
    text_features_list = []

    with torch.no_grad():
        for i, (image, instruction) in enumerate(
            zip(images, instructions, strict=True),
        ):
            vision_features = encode_image(
                image=image,
                model=vision_model,
                device=device,
            )
            vision_features_list.append(vision_features)

            tokens = tokenizer(
                instruction,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            )
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)

            outputs = language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            text_embedding = outputs.last_hidden_state.mean(dim=1)
            text_features_list.append(text_embedding)

            if (i + 1) % 100 == 0:
                logger.info(f"Pre-computed {i + 1}/{len(images)} samples")

    return vision_features_list, text_features_list


def compute_context_batch(
    batch_indices: list[int],
    vision_features_list: list[Tensor],
    text_features_list: list[Tensor],
    vision_projection: VisionProjection,
) -> Tensor:
    context_list = []

    for idx in batch_indices:
        vision_features = vision_features_list[idx]
        text_pooled = text_features_list[idx]

        vision_tokens = vision_projection(vision_features)
        vision_pooled = vision_tokens.mean(dim=1)

        context = torch.cat([vision_pooled, text_pooled], dim=-1)
        context_list.append(context)

    return torch.cat(context_list, dim=0)
