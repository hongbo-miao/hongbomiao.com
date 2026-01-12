import logging
import re
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-0.6B"
DATASET_NAME = "openai/gsm8k"
OUTPUT_DIRECTORY = Path("output")


SYSTEM_PROMPT = """You are a helpful math tutor. Solve the given math problem step by step.
Show your reasoning clearly, then provide the final numerical answer in the format: #### NUMBER

Example:
Question: If John has 5 apples and buys 3 more, how many apples does he have?
Answer: John starts with 5 apples. He buys 3 more apples. So he has 5 + 3 = 8 apples in total.
#### 8"""


def build_messages(question: str) -> list[dict[str, str]]:
    """Build chat messages for a GSM8K question."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def extract_answer(text: str) -> str | None:
    """Extract final numerical answer from GSM8K format: #### NUMBER."""
    match = re.search(r"####\s*([-+]?\d+\.?\d*)", text)
    if match:
        return match.group(1)
    return None


def extract_ground_truth_answer(text: str) -> str:
    """Extract answer from GSM8K ground truth. Raises error if not found."""
    answer = extract_answer(text)
    if answer is None:
        msg = f"Ground truth missing #### format: {text[:100]}..."
        raise ValueError(msg)
    return answer


def compute_reward(
    *,
    completions: list[str],
    prompts: list[str],
    answer: list[str],
    **kwargs: Any,  # noqa: ANN401
) -> list[float]:
    """
    Compute rewards for generated completions with partial credit.

    GRPO requires reward variance within groups. This softer reward function
    provides variance even when the model hasn't learned the #### format yet:
    - 1.0: Correct answer with #### NUMBER format
    - 0.5: Correct number appears somewhere in text
    - -0.5: Has numbers but wrong answer
    - -1.0: No numbers at all
    """
    del prompts, kwargs  # Unused
    rewards = []
    for completion, ground_truth in zip(completions, answer, strict=True):
        expected_answer = extract_ground_truth_answer(ground_truth)
        predicted_answer = extract_answer(completion)

        if predicted_answer is not None and predicted_answer == expected_answer:
            # Correct answer with proper format
            rewards.append(1.0)
        elif expected_answer in completion:
            # Correct number appears somewhere in text (partial credit)
            rewards.append(0.5)
        elif re.search(r"\d+", completion):
            # Has numbers but wrong answer
            rewards.append(-0.5)
        else:
            # No numbers at all
            rewards.append(-1.0)
    return rewards


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

    # GRPO requires a dataset with prompts. The model generates multiple responses per prompt,
    # and a reward function scores each response. The algorithm then uses group-relative
    # advantages to optimize the policy.
    # GSM8K contains grade school math problems with step-by-step solutions and final answers.
    logger.info("Loading dataset")
    dataset = load_dataset(DATASET_NAME, "main", split="train[:1000]")

    def format_prompt(example: dict[str, str]) -> dict[str, str]:
        """
        Format prompt using chat template with thinking mode disabled.

        Qwen3 has thinking mode enabled by default, which generates <think> tokens
        indefinitely. Setting enable_thinking=False prevents this and ensures the
        model generates completions that can properly terminate with EOS.
        """
        messages = build_messages(example["question"])
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return {"prompt": prompt}

    # Format prompts with system instructions and few-shot example.
    # This teaches the model the expected output format (#### NUMBER).
    dataset = dataset.map(format_prompt, remove_columns=["question"])

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

    # GRPOConfig extends TrainingArguments with GRPO-specific options.
    # Group Relative Policy Optimization (GRPO) is an RL algorithm that optimizes language models
    # using group-relative advantages without requiring a critic model.
    #
    # For each prompt $x$, GRPO:
    # 1. Samples a group of $G$ responses $\{y_1, ..., y_G\}$ from the current policy $\pi_\theta$
    # 2. Computes rewards $\{r_1, ..., r_G\}$ using a reward function
    # 3. Computes group-relative advantages: $\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G) + \epsilon}$
    #    where $\epsilon$ (typically $10^{-4}$) prevents division by zero when all rewards are identical.
    # 4. Optimizes the clipped surrogate objective with KL regularization:
    #
    # $\mathcal{L}_\text{GRPO}(\theta) = \mathbb{E}_{x \sim D, y_i \sim \pi_\theta(y|x)} \left[ \min\left( \rho_i \hat{A}_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta \cdot D_\text{KL}(\pi_\theta \| \pi_\text{ref}) \right]$
    #
    # where:
    # - $\rho_i = \frac{\pi_\theta(y_i|x)}{\pi_\text{old}(y_i|x)}$ is the probability ratio
    # - $D$ is the dataset of prompts
    # - $\pi_\theta$ is the policy being trained
    # - $\pi_\text{old}$ is the policy before the current update step
    # - $\pi_\text{ref}$ is the reference policy (frozen copy of the initial model)
    # - $\epsilon$ is the clipping parameter (typically 0.1-0.2)
    # - $\beta$ is the KL penalty coefficient controlling deviation from the reference
    # - $D_\text{KL}(\pi_\theta \| \pi_\text{ref})$ is the KL divergence measuring how much
    #   the current policy has drifted from the reference policy, computed per-token as:
    #   $D_\text{KL}(\pi_\theta \| \pi_\text{ref}) = \sum_t \pi_\theta(y_t|x, y_{<t}) \log \frac{\pi_\theta(y_t|x, y_{<t})}{\pi_\text{ref}(y_t|x, y_{<t})}$
    #
    training_config = GRPOConfig(
        output_dir=str(OUTPUT_DIRECTORY),
        # Number of responses $G$ to sample per prompt for computing group-relative advantages.
        # Larger groups provide more stable advantage estimates but require more compute.
        # Common values: 4, 8, 16. Memory scales linearly with this value.
        num_generations=8,
        # Maximum sequence length $T$ for generation. Generated tokens beyond this are truncated.
        # GSM8K requires step-by-step reasoning, so we need sufficient length for the model
        # to show its work before producing the final answer.
        max_completion_length=512,
        # Micro-batch size $b$ per GPU. This is the number of prompts processed together.
        # Actual samples = batch_size * num_generations per step.
        per_device_train_batch_size=1,
        # Accumulate gradients over $k$ steps before updating weights.
        # Effective batch size: $B_{eff} = b \times k \times n_{gpu}$
        gradient_accumulation_steps=8,
        # Number of epochs $E$ (complete passes through the dataset).
        num_train_epochs=1,
        # Learning rate $\eta$ for AdamW optimizer.
        # For smaller models (< 1B params), slightly higher learning rates work better.
        # Common range: $1 \times 10^{-5}$ to $2 \times 10^{-5}$ for small models.
        learning_rate=1e-5,
        # Use bfloat16 (brain floating point): 1 sign + 8 exponent + 7 mantissa bits.
        # Same dynamic range as fp32 but less precision. Faster and less memory than fp32.
        bf16=True,
        # KL penalty coefficient $\beta$ controlling deviation from reference policy.
        # TRL defaults to $\beta = 0.0$. This works well when the reward function is
        # well-defined (e.g., exact match) and using LoRA which limits drift.
        # Set to non-zero (e.g., 0.01-0.04) if you observe degenerate outputs.
        beta=0.0,
        # Gradient checkpointing trades compute for memory by recomputing activations during backward pass.
        gradient_checkpointing=False,
        # Temperature $\tau$ for sampling during generation. Higher values increase diversity.
        # $P(token) \propto \exp(\text{logit} / \tau)$
        # Lower temperature (0.7-0.9) for more focused outputs, higher (1.0-1.2) for exploration.
        temperature=0.9,
        # Log training metrics every N steps.
        logging_steps=10,
        # Save model checkpoint every N steps.
        save_steps=100,
        # Where to report metrics.
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        reward_funcs=compute_reward,
    )

    logger.info("Starting GRPO training")
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
