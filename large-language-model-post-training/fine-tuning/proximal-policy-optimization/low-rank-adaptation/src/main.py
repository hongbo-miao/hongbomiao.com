import logging
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import PPOConfig, PPOTrainer

logger = logging.getLogger(__name__)

POLICY_MODEL_NAME = "Qwen/Qwen3-0.6B"
REWARD_MODEL_NAME = "Skywork/Skywork-Reward-V2-Qwen3-8B"
DATASET_NAME = "trl-lib/tldr"
OUTPUT_DIRECTORY = Path("output")


def main() -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )

    logger.info(f"Loading tokenizer: {POLICY_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_NAME, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the policy model (the model we're training to generate better responses).
    # This is an autoregressive language model that generates text given a prompt.
    logger.info(f"Loading policy model: {POLICY_MODEL_NAME}")
    policy_model = AutoModelForCausalLM.from_pretrained(
        POLICY_MODEL_NAME,
        dtype=torch.bfloat16,
    ).to(device)

    # Load the reward model (scores how good a response is).
    # In RLHF, the reward model is typically trained on human preference data
    # to predict which response humans would prefer.
    # Skywork-Reward-V2-Qwen3-8B is pre-trained on human preferences (score head is trained).
    logger.info(f"Loading reward model: {REWARD_MODEL_NAME}")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        num_labels=1,
    ).to(device)

    # Load the value model (estimates expected future rewards from a state).
    # In PPO, the value model (critic) helps reduce variance in policy gradient estimates
    # by providing a baseline for advantage computation.
    # Value model uses smaller base model; its head is randomly initialized but learns during training.
    logger.info(f"Loading value model: {POLICY_MODEL_NAME}")
    value_model = AutoModelForSequenceClassification.from_pretrained(
        POLICY_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        num_labels=1,
    ).to(device)

    # Load the reference model (frozen copy of the initial policy).
    # Used to compute KL divergence penalty to prevent the policy from
    # deviating too far from the original model.
    logger.info(f"Loading reference model: {POLICY_MODEL_NAME}")
    reference_model = AutoModelForCausalLM.from_pretrained(
        POLICY_MODEL_NAME,
        torch_dtype=torch.bfloat16,
    ).to(device)

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
        # Other architectures may use diffeerent names (e.g., "fc1", "fc2" for standard MLP).
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

    # PPOConfig configures the Proximal Policy Optimization algorithm.
    # PPO is an actor-critic RL algorithm that optimizes a clipped surrogate objective
    # to ensure stable policy updates.
    #
    # The PPO objective function:
    #
    # $\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$
    #
    # where:
    # - $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio
    # - $\hat{A}_t$ is the estimated advantage (computed using GAE: Generalized Advantage Estimation)
    # - $\epsilon$ is the clipping parameter (typically 0.1-0.2)
    #
    # GAE computes advantages as:
    #
    # $\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$
    #
    # where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD residual.
    #
    # The total loss combines policy loss, value loss, and KL penalty:
    #
    # $\mathcal{L} = \mathcal{L}^{CLIP} - c_1 \mathcal{L}^{VF} - \beta D_{KL}(\pi_\theta || \pi_{ref})$
    #
    training_config = PPOConfig(
        output_dir=str(OUTPUT_DIRECTORY),
        # Learning rate $\eta$ for AdamW optimizer.
        learning_rate=1e-5,
        # Micro-batch size $b$ per GPU before gradient update.
        per_device_train_batch_size=1,
        # Accumulate gradients over $k$ steps before updating weights.
        gradient_accumulation_steps=16,
        # Total number of episodes (prompt-response pairs) to train on.
        total_episodes=1000,
        # Maximum length of generated responses.
        response_length=128,
        # Number of optimization epochs per batch of experience.
        # More epochs = more updates per batch, but risk of overfitting to current batch.
        num_ppo_epochs=4,
        # Discount factor $\gamma$ for future rewards in GAE.
        # $\gamma$ closer to 1 values future rewards more.
        gamma=0.99,
        # GAE lambda $\lambda$ for bias-variance tradeoff in advantage estimation.
        # $\lambda = 1$: high variance, low bias (Monte Carlo)
        # $\lambda = 0$: low variance, high bias (TD(0))
        lam=0.95,
        # Clipping parameter $\epsilon$ for policy ratio.
        # Prevents large policy updates by clipping the ratio to $[1-\epsilon, 1+\epsilon]$.
        cliprange=0.2,
        # Clipping range for value function updates.
        cliprange_value=0.2,
        # Coefficient for value function loss.
        vf_coef=0.1,
        # KL penalty coefficient $\beta$ controlling deviation from reference policy.
        kl_coef=0.05,
        # Use bfloat16 for training.
        bf16=True,
        # Log training metrics every N steps.
        logging_steps=10,
        # Save model checkpoint every N steps.
        save_steps=100,
        # Where to report metrics. Options: "wandb", "tensorboard", "none".
        report_to="none",
        # Penalty for responses that don't end with EOS token.
        missing_eos_penalty=1.0,
    )

    # PPO requires a dataset with prompts. The trainer handles:
    # 1. Sampling prompts from the dataset
    # 2. Generating responses using the current policy
    # 3. Computing rewards using the reward model
    # 4. Updating the policy using PPO with the collected experience
    logger.info("Loading dataset")
    dataset = load_dataset(DATASET_NAME, split="train[:1000]")
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(examples["prompt"], padding=False, truncation=True),
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Split dataset into train and eval sets with shuffling to avoid ordering bias.
    evaluation_sample_count = 100
    split_datasets = tokenized_dataset.train_test_split(
        test_size=evaluation_sample_count,
        seed=42,
    )
    train_dataset = split_datasets["train"]
    eval_dataset = split_datasets["test"]

    trainer = PPOTrainer(
        args=training_config,
        processing_class=tokenizer,
        model=policy_model,
        ref_model=reference_model,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
    )

    logger.info("Starting PPO training")
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
