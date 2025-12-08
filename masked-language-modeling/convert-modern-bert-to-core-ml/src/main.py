import logging
from pathlib import Path

import coremltools as ct
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers.modeling_outputs import MaskedLMOutput
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class ModernBertMaskedLanguageModelWrapper(torch.nn.Module):
    def __init__(self, hugging_face_model: AutoModelForMaskedLM) -> None:
        super().__init__()
        self.hugging_face_model = hugging_face_model

    def forward(
        self,
        input_id_tensor: torch.Tensor,
        attention_mask_tensor: torch.Tensor,
    ) -> torch.Tensor:
        model_output: MaskedLMOutput = self.hugging_face_model(
            input_ids=input_id_tensor,
            attention_mask=attention_mask_tensor,
        )
        return model_output.logits


def main() -> None:
    model_identifier: str = "answerdotai/ModernBERT-base"
    sequence_length: int = 128
    sample_sentence: str = "The [MASK] of [MASK] is Paris."
    output_directory: Path = Path("output")
    output_directory.mkdir(parents=True, exist_ok=True)

    logger.info("Loading ModernBERT tokenizer and model from Hugging Face Hub")
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_identifier)
    modern_bert_model: AutoModelForMaskedLM = AutoModelForMaskedLM.from_pretrained(
        model_identifier,
    )
    modern_bert_model.eval()

    logger.info(
        f"Preparing padded sample inputs with a maximum length of {sequence_length} tokens",
    )
    tokenized_inputs: BatchEncoding = tokenizer(
        sample_sentence,
        padding="max_length",
        max_length=sequence_length,
        return_tensors="pt",
    )
    input_id_tensor: torch.Tensor = tokenized_inputs["input_ids"]
    attention_mask_tensor: torch.Tensor = tokenized_inputs["attention_mask"]

    logger.info("Tracing ModernBERT for Core ML conversion")
    wrapped_model = ModernBertMaskedLanguageModelWrapper(modern_bert_model)
    traced_model: torch.jit.ScriptModule = torch.jit.trace(
        wrapped_model,
        (input_id_tensor, attention_mask_tensor),
    )
    traced_model.eval()

    logger.info("Converting traced graph to Core ML format")
    core_ml_model = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=input_id_tensor.shape,
                dtype=int,
            ),
            ct.TensorType(
                name="attention_mask",
                shape=attention_mask_tensor.shape,
                dtype=int,
            ),
        ],
        outputs=[ct.TensorType(name="logits")],
        compute_units=ct.ComputeUnit.ALL,
    )
    core_ml_model.short_description = (
        "ModernBERT-base masked language model converted for iOS inference"
    )
    core_ml_model.input_description["input_ids"] = (
        "Token identifiers padded to the configured maximum length"
    )
    core_ml_model.input_description["attention_mask"] = (
        "Attention mask aligned with the padded token identifiers"
    )
    core_ml_model.output_description["logits"] = (
        "Masked language model logits for each token position"
    )

    core_ml_model_path: Path = output_directory / "ModernBERTMaskedLM.mlpackage"
    logger.info(f"Saving Core ML package to {core_ml_model_path}")
    core_ml_model.save(str(core_ml_model_path))

    modern_bert_tokenizer_directory: Path = output_directory / "ModernBERTTokenizer"
    logger.info(f"Saving tokenizer assets to {modern_bert_tokenizer_directory}")
    tokenizer.save_pretrained(modern_bert_tokenizer_directory)

    logger.info("ModernBERT conversion finished successfully")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
