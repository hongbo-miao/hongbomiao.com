# Direct Preference Optimization (DPO) - Low-Rank Adaptation (LoRA)

## Core Idea

Direct Preference Optimization (DPO) aligns language models with human preferences without training a separate reward model. Combined with Low-Rank Adaptation (LoRA), it enables efficient preference learning by training low-rank update matrices instead of modifying all model weights.

### DPO Objective

DPO directly optimizes the policy using preference data. Given a prompt $x$, a preferred response $y_w$ (winner), and a rejected response $y_l$ (loser), DPO minimizes:

```math
\mathcal{L}_\text{DPO}(\pi_\theta; \pi_\text{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)} \right) \right]
```

where:

- $D$ is the preference dataset containing triplets $(x, y_w, y_l)$ where humans indicated $y_w$ is preferred over $y_l$
- $\mathbb{E}_{(x, y_w, y_l) \sim D}$ is the expectation over samples drawn from $D$
- $\pi_\theta$ is the policy being trained
- $\pi_\text{ref}$ is the reference policy (frozen copy of the initial model)
- $\beta$ is the temperature parameter controlling deviation from the reference
- $\sigma$ is the sigmoid function

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

- Prepare preference pairs from the dataset (prompt, chosen response, rejected response)
- Tokenize both chosen and rejected responses with the same prompt
- Apply LoRA adapters to attention and MLP layers
- Compute log probabilities for both responses under policy and reference model
- Calculate DPO loss based on preference margins
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
