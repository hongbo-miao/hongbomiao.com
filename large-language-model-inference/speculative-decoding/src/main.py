import logging
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


def load_models() -> tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizer, str]:
    logger.info("Loading models...")
    # Large model (target) and small model (draft)
    target_model_name: str = "EleutherAI/gpt-neo-1.3B"
    draft_model_name: str = "EleutherAI/gpt-neo-125M"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(target_model_name)
    target_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        target_model_name,
    ).to(device)
    draft_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        draft_model_name,
    ).to(device)

    return target_model, draft_model, tokenizer, device


def generate_text_and_time(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    device: str,
    assistant_model: PreTrainedModel | None = None,
) -> tuple[str, float]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    start_time: float = time.time()

    outputs: torch.Tensor = model.generate(
        **inputs,
        assistant_model=assistant_model,
        max_new_tokens=50,
        do_sample=False,  # Deterministic for fair comparison
        pad_token_id=tokenizer.eos_token_id,
    )

    generation_time: float = time.time() - start_time

    # Extract only new tokens
    new_tokens: torch.Tensor = outputs[0, inputs.input_ids.shape[1] :]
    text: str = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return text, generation_time


def main() -> None:
    # Load models
    target_model, draft_model, tokenizer, device = load_models()

    # Test prompt
    prompt: str = "The future of artificial intelligence is"
    logger.info(f"Prompt: '{prompt}'")

    # Method 1: Normal generation
    logger.info("Normal Generation:")
    normal_text, normal_time = generate_text_and_time(
        target_model,
        tokenizer,
        prompt,
        device,
    )
    logger.info(f"Time: {normal_time:.2f}s")
    logger.info(f"Text: {normal_text}")

    # Method 2: Speculative decoding
    logger.info("Speculative Decoding:")
    spec_text, spec_time = generate_text_and_time(
        target_model,
        tokenizer,
        prompt,
        device,
        assistant_model=draft_model,
    )
    logger.info(f"Time: {spec_time:.2f}s")
    logger.info(f"Text: {spec_text}")

    # Compare performance
    speedup: float = normal_time / spec_time if spec_time > 0 else 0
    logger.info("Comparison:")
    logger.info(f"Normal:      {normal_time:.2f}s")
    logger.info(f"Speculative: {spec_time:.2f}s")
    logger.info(f"Speedup:     {speedup:.2f}x")

    if speedup > 1:
        logger.info("✅ Speculative decoding is faster!")
    else:
        logger.info("⚠️  Speculative decoding overhead detected")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
