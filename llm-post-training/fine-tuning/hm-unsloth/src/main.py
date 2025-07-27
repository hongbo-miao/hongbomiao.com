# https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(14B)-Reasoning-Conversational.ipynb

from unsloth import FastLanguageModel  # isort: skip
from unsloth.chat_templates import standardize_sharegpt  # isort: skip

import logging
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, TextStreamer
from transformers.trainer_utils import TrainOutput
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)


def load_model() -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load and configure the Qwen3-14B model."""
    logger.info("Loading Qwen3-14B model...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-14B",
        max_seq_length=2048,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    return model, tokenizer


def prepare_datasets(
    tokenizer: PreTrainedTokenizer,
    chat_percentage: float = 0.25,
) -> Dataset:
    """Load and prepare training datasets."""
    logger.info("Loading datasets...")

    # Load reasoning and non-reasoning datasets
    reasoning_dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
    non_reasoning_dataset = load_dataset("mlabonne/FineTome-100k", split="train")

    logger.info(f"Reasoning dataset size: {len(reasoning_dataset)}")
    logger.info(f"Non-reasoning dataset size: {len(non_reasoning_dataset)}")

    # Convert reasoning dataset to conversational format
    def generate_conversation(
        examples: dict[str, list[str]],
    ) -> dict[str, list[list[dict[str, str]]]]:
        problems = examples["problem"]
        solutions = examples["generated_solution"]
        conversations = []
        for problem, solution in zip(problems, solutions, strict=False):
            conversations.append(
                [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": solution},
                ],
            )
        return {"conversations": conversations}

    reasoning_conversations = tokenizer.apply_chat_template(
        reasoning_dataset.map(generate_conversation, batched=True)["conversations"],
        tokenize=False,
    )

    # Convert non-reasoning dataset to conversational format
    dataset = standardize_sharegpt(non_reasoning_dataset)
    non_reasoning_conversations = tokenizer.apply_chat_template(
        dataset["conversations"],
        tokenize=False,
    )

    # Sample non-reasoning dataset based on chat percentage
    non_reasoning_subset = pd.Series(non_reasoning_conversations)
    non_reasoning_subset = non_reasoning_subset.sample(
        int(len(reasoning_conversations) * (chat_percentage / (1 - chat_percentage))),
        random_state=2407,
    )

    logger.info(f"Reasoning conversations: {len(reasoning_conversations)}")
    logger.info(f"Non-reasoning subset: {len(non_reasoning_subset)}")
    logger.info(
        f"Chat percentage: {len(non_reasoning_subset) / (len(non_reasoning_subset) + len(reasoning_conversations))}",
    )

    # Combine datasets
    data = pd.concat(
        [
            pd.Series(reasoning_conversations),
            pd.Series(non_reasoning_subset),
        ],
    )
    data.name = "text"

    combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
    return combined_dataset.shuffle(seed=3407)


def train_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
) -> TrainOutput:
    """Train the model using SFTTrainer."""
    logger.info("Starting training...")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=30,  # Set num_train_epochs=1 for full training
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
        ),
    )

    # Show initial memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    logger.info(f"{start_gpu_memory} GB of memory reserved.")

    # Train the model
    trainer_stats = trainer.train()

    # Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

    logger.info(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    logger.info(
        f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.",
    )
    logger.info(f"Peak reserved memory = {used_memory} GB.")
    logger.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    logger.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
    logger.info(
        f"Peak reserved memory for training % of max memory = {lora_percentage} %.",
    )
    return trainer_stats


def test_inference(model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> None:
    """Test the trained model with inference examples."""
    logger.info("=" * 50)
    logger.info("Testing inference...")
    logger.info("=" * 50)

    # Test without thinking
    logger.info("\nTesting without thinking:")
    messages = [
        {"role": "user", "content": "Solve (x + 2)^2 = 0."},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    _ = model.generate(
        **tokenizer(text, return_tensors="pt").to("cuda"),
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )

    # Test with thinking
    logger.info("Testing with thinking:")
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    _ = model.generate(
        **tokenizer(text, return_tensors="pt").to("cuda"),
        max_new_tokens=1024,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )


def save_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_path: Path,
) -> None:
    """Save the trained model."""
    logger.info(f"Saving model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info("Model saved successfully!")


def main() -> None:
    logger.info("Qwen3 (14B) Reasoning Conversational Fine-tuning")
    logger.info("=" * 50)

    try:
        # Load model and tokenizer
        model, tokenizer = load_model()

        # Prepare datasets
        combined_dataset = prepare_datasets(tokenizer, chat_percentage=0.25)

        # Train the model
        trainer_stats = train_model(model, tokenizer, combined_dataset)
        logger.info(f"Training completed with metrics: {trainer_stats}")

        # Test inference
        test_inference(model, tokenizer)

        # Save the model
        output_dir = Path("output")
        save_model(model, tokenizer, output_dir)

        logger.info("=" * 50)
        logger.info("Training completed successfully!")
        logger.info("=" * 50)

    except Exception:
        logger.exception("Error occurred.")
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
