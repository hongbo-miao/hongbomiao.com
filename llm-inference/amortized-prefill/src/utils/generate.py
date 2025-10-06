import logging
import time

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
) -> None:
    for sample_idx in range(num_samples):
        start_sample: float = time.time()
        generated: torch.Tensor = input_ids.clone()
        past: tuple | None = None

        for _ in range(max_new_tokens):
            with torch.no_grad():
                out = model(
                    generated[:, -1:] if past else generated,
                    past_key_values=past,
                    use_cache=True,
                )
            logits: torch.Tensor = out.logits[:, -1, :]
            past = out.past_key_values

            probs: torch.Tensor = torch.softmax(logits / temperature, dim=-1)
            next_token: torch.Tensor = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

        end_sample: float = time.time()
        logger.info(f"\n=== Sample {sample_idx + 1} (no amortized prefill) ===")
        logger.info(tokenizer.decode(generated[0], skip_special_tokens=True))
        logger.info(
            f"Sample {sample_idx + 1} time: {end_sample - start_sample:.3f} sec",
        )
