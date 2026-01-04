import logging
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-0.6B"
OUTPUT_DIRECTORY = Path("output")


def main() -> None:
    logger.info(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading dataset")
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:1000]")
    formatted_dataset = dataset.map(
        lambda example: {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            ),
        },
        remove_columns=dataset.column_names,
    )

    # Low-Rank Adaptation (LoRA) decomposes weight updates into two smaller matrices,
    # dramatically reducing trainable parameters while preserving model quality.
    # Instead of updating full weight matrix $W \in \mathbb{R}^{d \times k}$, LoRA learns:
    #
    # $W' = W + \Delta W = W + \frac{\alpha}{r} BA$
    #
    # where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ are low-rank matrices.
    # This reduces trainable parameters from $d \times k$ to $r \times (d + k)$.
    lora_config = LoraConfig(
        # Rank $r$ of the low-rank matrices $A$ and $B$.
        # Trainable parameters per layer: $r \times (d_{in} + d_{out})$ instead of $d_{in} \times d_{out}$.
        # Higher $r$ = more parameters = more capacity but slower. Common values: 8, 16, 32, 64.
        r=32,
        # Scaling factor $\alpha$ for LoRA updates. The actual output is scaled by $\frac{\alpha}{r}$.
        # Final layer output: $h = W_0 x + \frac{\alpha}{r} \cdot BAx$
        # Here: $\frac{64}{32} = 2$ scaling. Rule of thumb: $\alpha = 2r$ is a reasonable starting point.
        lora_alpha=64,
        # Which layers to apply LoRA to. These names are specific to Qwen3 architecture.
        # Attention: $Q = W_q x$, $K = W_k x$, $V = W_v x$, $O = W_o \cdot \text{Attention}(Q,K,V)$
        # MLP (SwiGLU, used by Qwen3/Llama/Mistral): $\text{MLP}(x) = W_{down} \cdot (\text{SiLU}(W_{gate} x) \odot W_{up} x)$
        # Other architectures may use different names (e.g., "fc1", "fc2" for standard MLP).
        # More modules = more trainable params but better adaptation capability.
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        # Task type determines how LoRA is applied. CAUSAL_LM is for autoregressive language models.
        task_type="CAUSAL_LM",
    )

    # SFTConfig extends TrainingArguments with SFT-specific options.
    # Supervised Fine-Tuning (SFT) trains the model to minimize the cross-entropy loss:
    #
    # $\mathcal{L} = -\sum_{t=1}^{T} \log P(y_t | y_{<t}, x; \theta)$
    #
    # where $x$ is the input, $y_t$ is the target token at position $t$, and $\theta$ are model parameters.
    training_config = SFTConfig(
        output_dir=str(OUTPUT_DIRECTORY),
        # Maximum sequence length $T$ for tokenization. Sequences longer than this are truncated.
        # Memory scales as $O(T^2)$ for attention. 1024 is reasonable for most conversational data.
        max_length=1024,
        # Micro-batch size $b$ per GPU before gradient update. Larger = more stable gradients but more memory.
        per_device_train_batch_size=8,
        # Accumulate gradients over $k$ steps before updating weights.
        # Effective batch size: $B_{eff} = b \times k \times n_{gpu}$
        # Here: $8 \times 2 = 16$ effective batch size (per GPU).
        gradient_accumulation_steps=2,
        # Number of epochs $E$ (complete passes through the dataset).
        # 1 epoch is often enough for fine-tuning; more epochs risk overfitting.
        num_train_epochs=1,
        # Learning rate $\eta$ for AdamW optimizer. Update rule: $\theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$
        # $2 \times 10^{-4}$ is typical for LoRA. Too high = unstable, too low = slow convergence.
        learning_rate=2e-4,
        # Use bfloat16 (brain floating point): 1 sign + 8 exponent + 7 mantissa bits.
        # Same dynamic range as fp32 but less precision. Faster and less memory than fp32.
        bf16=True,
        # Gradient checkpointing trades compute for memory by recomputing activations during backward pass.
        # Reduces memory from $O(L)$ to $O(\sqrt{L})$ where $L$ is number of layers. Slows training by ~20%.
        gradient_checkpointing=False,
        # Log training metrics (loss, learning rate, etc.) every N steps.
        logging_steps=10,
        # Save model checkpoint every N steps. Lower = more checkpoints but more disk usage.
        save_steps=100,
        # Where to report metrics. Options: "wandb", "tensorboard", "none".
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    logger.info("Starting training")
    trainer.train()

    logger.info(f"Saving model to {OUTPUT_DIRECTORY}")
    trainer.save_model(str(OUTPUT_DIRECTORY))
    tokenizer.save_pretrained(str(OUTPUT_DIRECTORY))
    logger.info("Done")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
