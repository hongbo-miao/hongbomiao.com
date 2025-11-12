import logging

from mlx_lm import generate, load

logger = logging.getLogger(__name__)


def main() -> None:
    model, tokenizer = load("Qwen/Qwen3-0.6B-MLX-4bit")
    prompt = "Hello, please introduce yourself and tell me what you can do."

    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        verbose=True,
        max_tokens=1024,
    )
    logger.info(response)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
