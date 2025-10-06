import logging
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from utils.generate import generate
from utils.generate_with_amortized_prefill import generate_with_amortized_prefill

logger = logging.getLogger(__name__)


def main() -> None:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()

    prompt: str = (
        "Here is a long story about airplanes and their adventures. "
        "Planes fly around the world, experiencing many funny incidents "
        "and challenges. Now, tell me exactly two jokes about airplanes. "
        "Only output the jokes, numbered, nothing else."
    )
    input_ids: torch.Tensor = tokenizer(prompt, return_tensors="pt").input_ids.to(
        model.device,
    )

    num_samples: int = 3
    max_new_tokens: int = 100
    temperature: float = 0.8

    start_no_amortized: float = time.time()
    generate(
        model,
        tokenizer,
        input_ids,
        num_samples,
        max_new_tokens,
        temperature,
    )
    end_no_amortized: float = time.time()
    logger.info(
        f"Total time (no amortized prefill): {end_no_amortized - start_no_amortized:.3f} sec",
    )

    start_amortized: float = time.time()
    generate_with_amortized_prefill(
        model,
        tokenizer,
        input_ids,
        num_samples,
        max_new_tokens,
        temperature,
    )
    end_amortized: float = time.time()
    logger.info(
        f"Total time (with amortized prefill): {end_amortized - start_amortized:.3f} sec",
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
