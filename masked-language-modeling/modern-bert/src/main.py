import logging

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers.modeling_outputs import MaskedLMOutput
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def main() -> None:
    model_identifier: str = "answerdotai/ModernBERT-base"
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_identifier)
    model: AutoModelForMaskedLM = AutoModelForMaskedLM.from_pretrained(model_identifier)

    input_text: str = "The [MASK] of [MASK] is Paris."
    tokenized_inputs: BatchEncoding = tokenizer(input_text, return_tensors="pt")

    with torch.no_grad():
        model_output: MaskedLMOutput = model(**tokenized_inputs)

    mask_position_tensor: torch.Tensor = (
        tokenized_inputs["input_ids"][0] == tokenizer.mask_token_id
    ).nonzero(as_tuple=True)[0]

    for mask_index, mask_position_value in enumerate(mask_position_tensor):
        mask_position: int = mask_position_value.item()
        token_logits: torch.Tensor = model_output.logits[0, mask_position]
        top_prediction_result = torch.topk(token_logits, k=5)
        predicted_tokens: list[str] = [
            tokenizer.decode([predicted_token_id])
            for predicted_token_id in top_prediction_result.indices
        ]
        predicted_scores: list[float] = top_prediction_result.values.softmax(
            dim=0,
        ).tolist()
        logger.info(f"[MASK] {mask_index}:")
        for predicted_token, predicted_score in zip(
            predicted_tokens,
            predicted_scores,
            strict=False,
        ):
            logger.info(f"  {predicted_token} ({predicted_score:.3%})")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
