import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def main() -> None:
    model_identifier: str = "Qwen/Qwen3-0.6B"
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_identifier)
    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(model_identifier)
    model.eval()

    input_text: str = "The meaning of life is"
    tokenized_inputs: BatchEncoding = tokenizer(input_text, return_tensors="pt")

    # Generate next tokens
    with torch.inference_mode():
        model_output: CausalLMOutputWithPast = model(**tokenized_inputs)

    # Get logits for the next token prediction (last position)
    next_token_logits: torch.Tensor = model_output.logits[0, -1, :]

    # Get top 5 predictions
    next_token_probs: torch.Tensor = torch.nn.functional.softmax(
        next_token_logits,
        dim=-1,
    )
    top_probs, top_indices = torch.topk(next_token_probs, k=5)
    predicted_tokens: list[str] = [
        tokenizer.decode([token_id]) for token_id in top_indices
    ]
    predicted_scores: list[float] = top_probs.tolist()

    logger.info(f"Input: '{input_text}'")
    logger.info("Top 5 next token predictions:")
    for predicted_token, predicted_score in zip(
        predicted_tokens,
        predicted_scores,
        strict=True,
    ):
        logger.info(f"  '{predicted_token}' ({predicted_score:.3%})")

    # Generate a full sequence
    logger.info("Generating full sequence:")
    generated_ids = model.generate(
        **tokenized_inputs,
        max_new_tokens=100,
        do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_text: str = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    logger.info(generated_text)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
