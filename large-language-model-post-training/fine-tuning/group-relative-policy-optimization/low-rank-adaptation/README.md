# Group Relative Policy Optimization (GRPO) - Low-Rank Adaptation (LoRA)

## Core Idea

Group Relative Policy Optimization (GRPO) is a reinforcement learning algorithm that optimizes language models using group-relative advantages without requiring a separate critic model. Combined with Low-Rank Adaptation (LoRA), it enables efficient policy optimization by training low-rank update matrices instead of modifying all model weights.

### GRPO Objective

For each prompt $x$, GRPO samples a group of $G$ responses $\{y_1, ..., y_G\}$ from the current policy $\pi_\theta$ and computes rewards $\{r_1, ..., r_G\}$ using a reward function. The group-relative advantage for each response is:

```math
\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G) + \epsilon}
```

where $\epsilon$ (typically $10^{-4}$) prevents division by zero when all rewards in a group are identical.

GRPO then optimizes a clipped surrogate objective with KL regularization:

```math
\mathcal{L}_\text{GRPO}(\theta) = \mathbb{E}_{x \sim D, y_i \sim \pi_\theta(y|x)} \left[ \min\left( \rho_i \hat{A}_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta \cdot D_\text{KL}(\pi_\theta \| \pi_\text{ref}) \right]
```

where:

- $\rho_i = \frac{\pi_\theta(y_i|x)}{\pi_\text{old}(y_i|x)}$ is the probability ratio
- $D$ is the dataset of prompts
- $\pi_\theta$ is the policy being trained
- $\pi_\text{old}$ is the policy before the current update step
- $\pi_\text{ref}$ is the reference policy (frozen copy of the initial model)
- $\epsilon$ is the clipping parameter (typically 0.1-0.2)
- $\beta$ is the KL penalty coefficient controlling deviation from the reference
- $D_\text{KL}(\pi_\theta \| \pi_\text{ref})$ is the KL divergence measuring how much the current policy has drifted from the reference policy, computed per-token as:

```math
D_\text{KL}(\pi_\theta \| \pi_\text{ref}) = \sum_t \pi_\theta(y_t|x, y_{\lt t}) \log \frac{\pi_\theta(y_t|x, y_{\lt t})}{\pi_\text{ref}(y_t|x, y_{\lt t})}
```

The KL penalty serves as a regularizer that prevents the policy from diverging too far from the reference model, which helps maintain response quality and prevents reward hacking.

> **Note:** TRL defaults to $\beta = 0.0$, meaning the KL penalty is disabled by default. This works well when:
>
> - The reward function is well-defined and hard to game (e.g., exact match on answers)
> - The task is constrained with clear correct/incorrect outcomes
> - Using LoRA, which already limits drift from the base model
>
> Set $\beta > 0$ (e.g., 0.01-0.04) if you observe degenerate outputs, reward increasing but quality decreasing, or loss of coherence.

### Key Differences from PPO

- No critic model needed (uses group statistics instead of value function)
- Advantages are computed relative to the group, not using Generalized Advantage Estimation (GAE)
- More memory efficient for large language model training
- Simpler implementation with fewer hyperparameters

### LoRA Decomposition

Instead of updating the full weight matrix $W \in \mathbb{R}^{d \times k}$, LoRA learns two smaller matrices:

- $A \in \mathbb{R}^{r \times k}$ (down-projection)
- $B \in \mathbb{R}^{d \times r}$ (up-projection)

where $r \ll \min(d, k)$ is the rank.

The effective weight becomes:

```math
W' = W + \Delta W = W + \frac{\alpha}{r} BA
```

where $\alpha$ is a scaling factor and $r$ is the rank.

### Training Process

In code:

- Load prompts from the dataset
- Apply LoRA adapters to attention and MLP layers
- For each prompt, generate $G$ responses from the current policy
- Compute rewards for each response using the reward function
- Calculate group-relative advantages using mean and standard deviation
- Compute clipped surrogate loss with KL penalty
- Backpropagate gradients through LoRA parameters (base model frozen)
- Update LoRA weights with optimizer (e.g., AdamW)

### Parameter Efficiency

LoRA drastically reduces trainable parameters:

```math
\text{LoRA params} = 2 \times r \times d \times n_\text{layers} \times n_\text{matrices}
```

where:

- $r$ = rank - controls adaptation capacity
- $d$ = hidden dimension of the model
- $n_\text{layers}$ = number of transformer layers
- $n_\text{matrices}$ = number of projection matrices per layer
- Factor of 2 = matrices $A$ and $B$ in each LoRA adapter

For Qwen3-0.6B (used in this project) with $r = 32$, $d = 1024$, 28 layers, and 7 target modules ($W_q$, $W_k$, $W_v$, $W_o$, $W_\text{gate}$, $W_\text{up}$, $W_\text{down}$):

```math
\text{Full fine-tuning: } \sim 600\text{M params}
```

```math
\text{LoRA: } 2 \times 32 \times 1024 \times 28 \times 7 \approx 12.8\text{M params} \approx 2.1\%
```

### Inference

After training, LoRA weights can be:

1. **Merged**: $W' = W + \frac{\alpha}{r} BA$ for zero inference overhead
2. **Kept separate**: Swap adapters for different tasks on the same base model
