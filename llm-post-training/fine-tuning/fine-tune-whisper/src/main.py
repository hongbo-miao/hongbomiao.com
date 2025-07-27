"""Whisper Fine-tuning for Multilingual ASR."""

import logging
from dataclasses import dataclass
from typing import Any

import evaluate
import torch
from datasets import Audio, Dataset, load_dataset
from transformers import (
    EvalPrediction,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME: str = "openai/whisper-tiny"
LANGUAGE: str = "French"
LANGUAGE_CODE: str = "fr"
DATASET_NAME: str = "mozilla-foundation/common_voice_17_0"
OUTPUT_DIR: str = "./output/whisper-tiny-fr"

# Dataset size parameters - None means use full dataset
TRAIN_SUBSET_SIZE: int | None = 10000  # Set to None for full dataset
TEST_SUBSET_SIZE: int | None = 2000  # Set to None for full dataset


@dataclass
class DataCollator:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(
        self,
        features: list[dict[str, list[int] | torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        # Pad input features
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        # Pad labels
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

        # Mask padding tokens in labels
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100,
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def load_and_prepare_dataset(
    train_subset_size: int | None = None,
    test_subset_size: int | None = None,
) -> tuple[Dataset, Dataset]:
    """Load Common Voice dataset with optional subset size."""
    logger.info(f"Loading dataset: {DATASET_NAME}")
    train_dataset = load_dataset(
        DATASET_NAME,
        LANGUAGE_CODE,
        split="train+validation",
        trust_remote_code=True,
    )
    test_dataset = load_dataset(
        DATASET_NAME,
        LANGUAGE_CODE,
        split="test",
        trust_remote_code=True,
    )

    # Apply subset selection if specified
    if train_subset_size is not None:
        original_train_size = len(train_dataset)
        subset_size = min(train_subset_size, original_train_size)
        train_dataset = train_dataset.select(range(subset_size))
        logger.info(f"Using train subset: {subset_size}/{original_train_size} samples")
    else:
        logger.info(f"Using full train dataset: {len(train_dataset)} samples")

    if test_subset_size is not None:
        original_test_size = len(test_dataset)
        subset_size = min(test_subset_size, original_test_size)
        test_dataset = test_dataset.select(range(subset_size))
        logger.info(f"Using test subset: {subset_size}/{original_test_size} samples")
    else:
        logger.info(f"Using full test dataset: {len(test_dataset)} samples")

    # Remove unnecessary columns
    cols_to_remove: list[str] = [
        "accent",
        "age",
        "client_id",
        "down_votes",
        "gender",
        "locale",
        "path",
        "segment",
        "up_votes",
    ]
    train_dataset = train_dataset.remove_columns(cols_to_remove)
    test_dataset = test_dataset.remove_columns(cols_to_remove)

    # Set audio sampling rate
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

    return train_dataset, test_dataset


def preprocess_function(
    batch: dict[str, Any],
    processor: WhisperProcessor,
) -> dict[str, Any]:
    """Preprocess audio and text."""
    audio = batch["audio"]

    # Extract audio features
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    ).input_features[0]

    # Tokenize text
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids

    return batch


def compute_metrics(
    pred: EvalPrediction,
    processor: WhisperProcessor,
    metric: evaluate.EvaluationModule,
) -> dict[str, float]:
    """Compute WER metric."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and references
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def main() -> None:
    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        logger.info("Using CPU - training will be slow!")

    # Load dataset with subset parameters
    train_dataset, test_dataset = load_and_prepare_dataset(
        train_subset_size=TRAIN_SUBSET_SIZE,
        test_subset_size=TEST_SUBSET_SIZE,
    )

    # Setup processor and model
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME,
        language=LANGUAGE,
        task="transcribe",
    )
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    model.generation_config.language = LANGUAGE_CODE
    model.generation_config.task = "transcribe"

    # Preprocess datasets
    logger.info("Preprocessing datasets...")
    train_dataset = train_dataset.map(
        lambda batch: preprocess_function(batch, processor),
        remove_columns=train_dataset.column_names,
        num_proc=2,
    )
    test_dataset = test_dataset.map(
        lambda batch: preprocess_function(batch, processor),
        remove_columns=test_dataset.column_names,
        num_proc=2,
    )

    # Setup data collator and metrics
    data_collator = DataCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    metric = evaluate.load("wer")

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=64,  # Much larger batches
        per_device_eval_batch_size=32,  # Increased eval batch
        gradient_accumulation_steps=1,  # No need with large batches
        learning_rate=2e-5,  # Slightly higher LR for larger batches
        warmup_steps=200,  # Reduced warmup for faster training
        max_steps=2000,  # Fewer steps needed with larger batches
        fp16=False,
        bf16=True,  # Use bfloat16 for better precision
        eval_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        ddp_find_unused_parameters=False,
        # Maximize CPU/GPU utilization
        dataloader_num_workers=16,  # Use more CPU cores
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,  # Prefetch more batches
        remove_unused_columns=False,
        report_to=None,
        # Memory optimizations for large GPU
        max_grad_norm=1.0,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
    )

    # Setup trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor, metric),
        processing_class=processor,
    )

    # Save processor and train
    processor.save_pretrained(OUTPUT_DIR)

    logger.info("Starting training...")
    trainer.train()

    logger.info("Training completed!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
