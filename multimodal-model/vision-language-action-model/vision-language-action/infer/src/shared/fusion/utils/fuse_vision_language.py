import logging

import torch
from torch import Tensor
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


def fuse_vision_language(
    vision_tokens: Tensor,
    text_embeddings: Tensor,
    language_model: PreTrainedModel,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """
    Fuse vision and language tokens through the language model.

    Concatenates projected vision tokens with text embeddings and passes
    through Qwen3 to get contextualized representations.

    Args:
        vision_tokens: Projected vision features [batch, num_patches, hidden_dim]
        text_embeddings: Text token embeddings [batch, seq_len, hidden_dim]
        language_model: Qwen3 model
        attention_mask: Optional attention mask for the fused sequence

    Returns:
        Contextualized embedding from the last hidden state [batch, total_seq_len, hidden_dim]

    """
    batch_size = vision_tokens.shape[0]
    num_vision_tokens = vision_tokens.shape[1]
    num_text_tokens = text_embeddings.shape[1]
    total_sequence_length = num_vision_tokens + num_text_tokens

    vision_tokens = vision_tokens.to(
        device=text_embeddings.device,
        dtype=text_embeddings.dtype,
    )

    logger.debug(
        f"Vision tokens shape: {vision_tokens.shape}, dtype: {vision_tokens.dtype}",
    )
    logger.debug(
        f"Text embeddings shape: {text_embeddings.shape}, dtype: {text_embeddings.dtype}",
    )

    fused_embeddings = torch.cat([vision_tokens, text_embeddings], dim=1)

    if attention_mask is None:
        attention_mask = torch.ones(
            batch_size,
            total_sequence_length,
            device=fused_embeddings.device,
            dtype=torch.long,
        )

    with torch.no_grad():
        outputs = language_model(
            inputs_embeds=fused_embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    if hasattr(outputs, "last_hidden_state"):
        context_embedding = outputs.last_hidden_state
    elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        context_embedding = outputs.hidden_states[-1]
    else:
        msg = "Cannot extract hidden states from model output"
        raise ValueError(msg)

    logger.debug(f"Fused context embedding shape: {context_embedding.shape}")

    return context_embedding
