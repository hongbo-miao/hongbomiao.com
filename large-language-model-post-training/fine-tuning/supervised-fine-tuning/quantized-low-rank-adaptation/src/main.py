import logging
from pathlib import Path

import torch
import torch.distributed
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-0.6B"
DATASET_NAME = "HuggingFaceH4/ultrachat_200k"
OUTPUT_DIRECTORY = Path("output")


def main() -> None:
    # For multi-GPU QLoRA, use DDP (DistributedDataParallel) via accelerate.
    # Each GPU loads its own copy of the 4-bit quantized model.
    device_string = PartialState().process_index

    # QLoRA (Quantized Low-Rank Adaptation) combines 4-bit quantization with LoRA.
    # The base model weights are quantized to 4-bit NormalFloat (NF4), reducing memory by ~4x,
    # while LoRA adapters remain in higher precision (bfloat16) for stable training.
    # Memory: 4-bit base (~0.5 bytes/param) + bf16 LoRA adapters + bf16 optimizer states for LoRA only.
    quantization_config = BitsAndBytesConfig(
        # Enable 4-bit quantization for loading model weights.
        # Reduces memory from 2 bytes/param (bf16) to ~0.5 bytes/param.
        load_in_4bit=True,
        # NF4 (4-bit NormalFloat) is optimized for normally distributed weights.
        # Quantization levels are spaced according to a normal distribution, matching typical weight distributions.
        # Alternative: "fp4" uses uniform spacing, slightly less accurate for neural network weights.
        bnb_4bit_quant_type="nf4",
        # Compute dtype for matrix multiplications during forward/backward pass.
        # Weights are dequantized to this dtype before computation: $W_{dequant} = \text{dequant}(W_{4bit})$
        # bfloat16 provides good balance of speed and numerical stability.
        bnb_4bit_compute_dtype=torch.bfloat16,
        # Double quantization: quantize the quantization constants themselves.
        # Saves ~0.4 bits per parameter with minimal accuracy loss.
        # The scaling factors $s$ are also quantized: $s_{quant} = \text{quant}(s)$
        bnb_4bit_use_double_quant=True,
    )

    logger.info(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        # For QLoRA with DDP, each process loads the model on its assigned GPU.
        # PartialState().process_index returns the local rank (0, 1, 2, ...) for each process.
        device_map={"": device_string},
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading dataset")
    dataset = load_dataset(DATASET_NAME, split="train_sft[:1000]")
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

    # QLoRA uses LoRA adapters on top of the 4-bit quantized base model.
    # The LoRA weight update: $W' = W_{4bit} + \frac{\alpha}{r} BA$
    # where $W_{4bit}$ is frozen and quantized, while $B$ and $A$ are trainable in bfloat16.
    # This achieves near full fine-tuning quality at a fraction of the memory cost.
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
        # Memory scales as $O(T^2)$ for attention. 512 is used for QLoRA to reduce memory usage.
        max_length=512,
        # Micro-batch size $b$ per GPU before gradient update. Reduced for QLoRA memory constraints.
        per_device_train_batch_size=2,
        # Accumulate gradients over $k$ steps before updating weights.
        # Effective batch size: $B_{eff} = b \times k \times n_{gpu}$
        # Here: $2 \times 8 = 16$ effective batch size (per GPU).
        gradient_accumulation_steps=8,
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
        # Essential for QLoRA to fit larger models in memory.
        gradient_checkpointing=True,
        # Required for gradient checkpointing with LoRA to avoid errors with reentrant checkpointing.
        gradient_checkpointing_kwargs={"use_reentrant": False},
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

    # Clean up distributed process group to avoid resource leak warning.
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
