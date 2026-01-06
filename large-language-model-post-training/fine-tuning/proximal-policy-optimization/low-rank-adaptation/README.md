# Proximal Policy Optimization (PPO) - Low-Rank Adaptation (LoRA)

## Core Idea

Proximal Policy Optimization (PPO) is an actor-critic reinforcement learning algorithm that optimizes language models using a clipped surrogate objective to ensure stable policy updates. Combined with Low-Rank Adaptation (LoRA), it enables efficient policy optimization by training low-rank update matrices instead of modifying all model weights.

### PPO Components

PPO for language model fine-tuning requires four models working together:

1. **Policy Model (Actor)**: The "player" being trained. It generates tokens given a prompt. Its weights are updated during training. Uses `Qwen/Qwen3-0.6B`.

2. **Value Model (Critic)**: The "referee/predictor". It estimates the expected cumulative reward from the current state, helping the actor judge whether a generated token is good or bad. Its weights are updated during training. Uses `Qwen/Qwen3-0.6B` with a randomly initialized score head. Note: The value model uses the smaller base model (not the reward model) because it learns to predict expected rewards during PPO training - it does not need pre-trained human preference knowledge. This also saves memory and is common practice in PPO implementations.

3. **Reward Model**: The "scorer" that mimics human preferences. Given a response, it returns a scalar score. It is frozen and does not participate in training. Uses `Skywork/Skywork-Reward-V2-Qwen3-8B` (pre-trained on human preferences).

4. **Reference Model**: The model's "original self" - a frozen copy of the policy before fine-tuning. Used to compute KL divergence, preventing the trained model from drifting too far and generating incoherent text just to chase high rewards. Uses `Qwen/Qwen3-0.6B`.

### PPO Objective

The core improvement of PPO is solving the problem where policy gradient algorithms take "steps that are too large", causing the model to collapse.

The PPO objective function uses a clipped surrogate to prevent large policy updates:

```math
\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
```

where:

- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio between new and old policy
  - $s_t$ (state): the prompt plus all tokens generated so far $(x, y_1, ..., y_{t-1})$
  - $a_t$ (action): the token generated at position $t$
- $\hat{A}_t$ is the estimated advantage (computed using GAE) - how much better this action is compared to average. If $\hat{A}_t > 0$, the action is good and we want to increase its probability.
- $\epsilon$ is the clipping parameter (typically 0.1-0.2)

**Why clipping?** If the difference between new and old policy exceeds $\epsilon$ (e.g., 20%), the loss function forcibly "clips" the gradient propagation. This ensures model updates are smooth and won't oscillate dramatically due to an anomalously high score.

### Generalized Advantage Estimation (GAE)

GAE finds a balance between **bias and variance** in advantage estimation:

```math
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
```

where:

- $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD (Temporal Difference) residual: current reward + discounted future predicted value - current predicted value
- $\gamma$ is the discount factor - how much to value future rewards
- $\lambda$ controls bias-variance tradeoff:
  - $\lambda = 1$: high variance, low bias (Monte Carlo - considers full trajectory)
  - $\lambda = 0$: low variance, high bias (TD(0) - only looks one step ahead)

**In LLM context**: The model must consider not only whether the current token is appropriate, but also how this token affects the logical flow of subsequent generation.

### Total Loss

The total loss combines three components:

```math
\mathcal{L} = \mathcal{L}^{CLIP} - c_1 \mathcal{L}^{VF} - \beta D_{KL}(\pi_\theta \| \pi_{ref})
```

where:

- $\mathcal{L}^{CLIP}$: Makes the model generate responses that humans prefer
- $\mathcal{L}^{VF}$ (Value Function Loss): Trains the critic model to predict more accurately (typically MSE loss). $c_1$ is its coefficient.
- $D_{KL}(\pi_\theta \| \pi_{ref})$ (KL Penalty): This is unique to LLM training. It measures the distribution difference between current model and initial model.

**Why KL penalty matters**: If the model starts generating strange but high-scoring gibberish just to please the reward model (known as **Reward Hacking**), the KL divergence increases, adding to the loss and penalizing the model, forcing it to maintain natural language.

### Training Workflow

The engineering flow in PPO training:

1. **Rollout (Sampling)**: Policy model generates a response given a prompt, saving the probability of each token.

2. **Evaluation (Scoring)**:

    2.1. Reward model scores the entire response
    2.2. Value model predicts scores at each step
    2.3. Reference model computes probability differences for KL

3. **Advantage Calculation**: Compute advantage value for each token using GAE.

4. **Optimization**:

    4.1. Backpropagate gradients through LoRA's $A$ and $B$ matrices (base model frozen)
    4.2. Update value model parameters

In code:

- Load policy model, value model, reward model, and reference model
- Apply LoRA adapters to attention and MLP layers of the policy model
- Load prompts from the dataset (trl-lib/tldr)
- For each prompt, generate responses using the current policy
- Compute rewards using the reward model (Skywork-Reward-V2-Qwen3-8B)
- Estimate advantages using GAE with the value model
- Compute clipped surrogate loss with KL penalty against reference model
- Backpropagate gradients through LoRA parameters (base model frozen)
- Update LoRA weights with optimizer (AdamW)

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
