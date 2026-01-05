import logging
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-0.6B"
DATASET_NAME = "trl-lib/ultrafeedback_binarized"
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

    # DPO requires a preference dataset with "chosen" and "rejected" responses.
    # In LLM context, "prompt" refers to the user's input/question that the model responds to.
    # The ultrafeedback_binarized dataset uses conversational format:
    # - "chosen": list of messages [{"role": "user", "content": prompt}, {"role": "assistant", "content": preferred_response}]
    # - "rejected": list of messages [{"role": "user", "content": prompt}, {"role": "assistant", "content": less_preferred_response}]
    # The DPOTrainer extracts the prompt from the first user message automatically.
    logger.info("Loading dataset")
    dataset = load_dataset(DATASET_NAME, split="train[:1000]")

    # Low-Rank Adaptation (LoRA) decomposes weight updates into two smaller matrices,
    # dramatically reducing trainable parameters while preserving model quality.
    # Instead of updating full weight matrix $W \in \mathbb{R}^{d \times k}$, LoRA learns:
    #
    # $W' = W + \Delta W = W + \frac{\alpha}{r} BA$
    #
    # where $A \in \mathbb{R}^{r \times k}$ (down-projection) and $B \in \mathbb{R}^{d \times r}$ (up-projection) are low-rank matrices.
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

    # DPOConfig extends TrainingArguments with DPO-specific options.
    # Direct Preference Optimization (DPO) directly optimizes the policy using preference data
    # without needing a separate reward model. The DPO loss is:
    #
    # $\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$
    #
    # where:
    # - $D$ is the preference dataset containing triplets $(x, y_w, y_l)$ where humans indicated $y_w$ is preferred over $y_l$
    # - $x$ is the prompt (user input/question)
    # - $y_w$ is the preferred (chosen) response
    # - $y_l$ is the rejected response
    # - $\pi_\theta$ is the policy being trained
    # - $\pi_{ref}$ is the reference policy (frozen copy of the initial model)
    # - $\beta$ is the temperature parameter controlling deviation from the reference
    # - $\sigma$ is the sigmoid function
    training_config = DPOConfig(
        output_dir=str(OUTPUT_DIRECTORY),
        # Maximum sequence length $T$ for tokenization. Sequences longer than this are truncated.
        # Memory scales as $O(T^2)$ for attention. 1024 is reasonable for most conversational data.
        max_length=1024,
        # Maximum length for the prompt portion. Prompts longer than this are truncated.
        max_prompt_length=512,
        # Micro-batch size $b$ per GPU before gradient update. Larger = more stable gradients but more memory.
        # DPO processes chosen and rejected pairs together, so effective samples = batch_size * 2.
        per_device_train_batch_size=4,
        # Accumulate gradients over $k$ steps before updating weights.
        # Effective batch size: $B_{eff} = b \times k \times n_{gpu}$
        # Here: $4 \times 4 = 16$ effective batch size (per GPU).
        gradient_accumulation_steps=4,
        # Number of epochs $E$ (complete passes through the dataset).
        # 1 epoch is often enough for fine-tuning; more epochs risk overfitting.
        num_train_epochs=1,
        # Learning rate $\eta$ for AdamW optimizer. Update rule: $\theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$
        # $5 \times 10^{-5}$ is typical for DPO (lower than SFT). Too high = unstable, too low = slow convergence.
        learning_rate=5e-5,
        # Use bfloat16 (brain floating point): 1 sign + 8 exponent + 7 mantissa bits.
        # Same dynamic range as fp32 but less precision. Faster and less memory than fp32.
        bf16=True,
        # Temperature parameter $\beta$ controlling deviation from reference policy.
        # Higher $\beta$ = stronger preference optimization but may deviate more from reference.
        # Lower $\beta$ = more conservative, stays closer to reference policy.
        # Common values: 0.1 to 0.5. Default is 0.1.
        beta=0.1,
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

    trainer = DPOTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    logger.info("Starting training")
    trainer.train()

    logger.info(f"Saving model to {OUTPUT_DIRECTORY}")
    trainer.save_model(str(OUTPUT_DIRECTORY))
    tokenizer.save_pretrained(str(OUTPUT_DIRECTORY))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
