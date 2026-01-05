# Supervised Fine-Tuning (SFT) - Low-Rank Adaptation (LoRA)

## Core Idea

Supervised Fine-Tuning (SFT) adapts a pre-trained language model to follow instructions by training on curated prompt-response pairs. Low-Rank Adaptation (LoRA) makes this process efficient by learning low-rank update matrices instead of modifying all model weights.

### LoRA Decomposition

Instead of updating the full weight matrix $W \in \mathbb{R}^{d \times k}$, LoRA learns two smaller matrices:

- $B \in \mathbb{R}^{d \times r}$ (down-projection)
- $A \in \mathbb{R}^{r \times k}$ (up-projection)

where $r \ll \min(d, k)$ is the rank.

The effective weight becomes:

```math
W' = W + \Delta W = W + \frac{\alpha}{r} BA
```

where $\alpha$ is a scaling factor and $r$ is the rank.

### Training Objective

SFT uses the standard language modeling cross-entropy loss on the response tokens:

```math
\mathcal{L}(\theta) = -\sum_{i=1}^{T} \log p_\theta(y_i \mid x, y_{\lt i})
```

where:

- $x$ is the instruction/prompt
- $y = (y_1, \dots, y_T)$ is the target response
- Only response tokens contribute to the loss (prompt tokens are masked)

### Training Process

In code:

- Prepare instruction-response pairs from the dataset
- Tokenize inputs with prompt template (e.g., `<|user|>\n{instruction}\n<|assistant|>\n{response}`)
- Apply LoRA adapters to attention layers (typically $W_q$, $W_k$, $W_v$, $W_o$)
- Forward pass through the model with LoRA weights
- Compute cross-entropy loss on response tokens only
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
