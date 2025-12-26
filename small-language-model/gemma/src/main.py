import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def main() -> None:
    model_id = "google/gemma-3-270m-it"

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )

    logger.info(f"Using device: {device}")

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model = model.to(device)

    messages = [
        {
            "role": "user",
            "content": "What is gravity?",
        },
    ]

    logger.info("Generating response...")
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info("Response: %s", response)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
