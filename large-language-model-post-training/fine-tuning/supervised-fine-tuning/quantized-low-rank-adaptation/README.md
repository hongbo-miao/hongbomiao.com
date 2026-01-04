# Supervised Fine-Tuning (SFT) - Quantized Low-Rank Adaptation (QLoRA)

## Core Idea

QLoRA combines 4-bit quantization with Low-Rank Adaptation (LoRA) to enable fine-tuning of large language models on consumer hardware. The base model weights are quantized to 4-bit NormalFloat (NF4), while LoRA adapters remain in higher precision (bfloat16) for stable training.

### QLoRA Memory Efficiency

QLoRA achieves ~4x memory reduction compared to LoRA:

|Component|LoRA|QLoRA|
|---|---|---|
|Base model|2 bytes/param (bf16)|~0.5 bytes/param (4-bit)|
|LoRA adapters|bf16|bf16|
|Optimizer states|Only for LoRA params|Only for LoRA params|

### 4-bit NormalFloat Quantization

NF4 quantization is optimized for normally distributed weights:

```math
W_{4bit} = \text{round}\left(\frac{W - \min(W)}{\max(W) - \min(W)} \times 15\right)
```

The quantization levels are spaced according to a normal distribution, matching typical neural network weight distributions. During forward/backward passes, weights are dequantized:

```math
W_{dequant} = \text{dequant}(W_{4bit})
```

### Double Quantization

QLoRA applies quantization to the quantization constants themselves, saving ~0.4 bits per parameter:

```math
s_{quant} = \text{quant}(s)
```

where $s$ are the scaling factors for each quantization block.

### LoRA on Quantized Weights

The LoRA weight update operates on the quantized base model:

```math
W' = W_{4bit} + \frac{\alpha}{r} BA
```

where:

- $W_{4bit}$ is frozen and quantized to 4-bit
- $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ are trainable in bfloat16
- $\alpha$ is the scaling factor, $r$ is the rank

### Training Objective

SFT uses the standard language modeling cross-entropy loss on the response tokens:

```math
\mathcal{L}(\theta) = -\sum_{i=1}^{T} \log p_\theta(y_i \mid x, y_{<i})
```

where:

- $x$ is the instruction/prompt
- $y = (y_1, \dots, y_T)$ is the target response
- Only response tokens contribute to the loss (prompt tokens are masked)

### Training Process

In code:

- Load base model with 4-bit quantization (BitsAndBytesConfig)
- Apply LoRA adapters to target modules (attention and MLP layers)
- Enable gradient checkpointing to reduce activation memory
- Forward pass: dequantize weights -> compute -> quantize gradients
- Backpropagate gradients through LoRA parameters only (base model frozen)
- Update LoRA weights with optimizer (e.g., AdamW)

### Hardware Requirements

QLoRA enables training that would otherwise require expensive hardware:

|Model Size|Full Fine-tuning|LoRA|QLoRA|
|---|---|---|---|
|7B|~120 GB|~28 GB|~6 GB|
|13B|~240 GB|~52 GB|~10 GB|
|70B|~1.2 TB|~280 GB|~48 GB|

### Key Configuration Parameters

For QLoRA training:

- `load_in_4bit=True`: Enable 4-bit quantization
- `bnb_4bit_quant_type="nf4"`: Use NormalFloat4 quantization
- `bnb_4bit_compute_dtype=torch.bfloat16`: Compute dtype for dequantized operations
- `bnb_4bit_use_double_quant=True`: Enable double quantization
- `gradient_checkpointing=True`: Essential for memory efficiency
- `device_map={"": device_string}`: For multi-GPU DDP, each process loads on its assigned GPU via `PartialState().process_index`

### Multi-GPU Training

QLoRA supports multi-GPU training via DDP (DistributedDataParallel) with accelerate.

Each GPU loads its own copy of the 4-bit quantized model. Note that DataParallel (nn.DataParallel) is not supported for quantized models.

### Why DDP Instead of Model Parallelism?

LoRA and QLoRA use different multi-GPU strategies due to quantization constraints:

#### LoRA: Model Parallelism

- Uses bf16 weights (2 bytes/param)
- Can use `device_map="auto"` which splits the model across GPUs
- One model instance with layers distributed across GPUs
- Activations flow between GPUs during forward/backward pass

```text
┌─────────────┐    ┌─────────────┐
│   GPU 0     │    │   GPU 1     │
│ Layers 0-15 │───>│ Layers 16-31│
│   (bf16)    │    │   (bf16)    │
└─────────────┘    └─────────────┘
      ONE MODEL SPLIT ACROSS GPUS
```

#### QLoRA: Data Parallelism (DDP)

- Quantized models (4-bit) cannot be split across GPUs
- `bitsandbytes` quantization requires each layer to reside on a single device
- Must use DDP via accelerate: each GPU loads its own complete copy of the quantized model
- Each GPU processes different data batches, gradients are synchronized after each step

```text
┌─────────────┐    ┌─────────────┐
│   GPU 0     │    │   GPU 1     │
│ Full Model  │    │ Full Model  │
│   (4-bit)   │    │   (4-bit)   │
│  Batch A    │    │  Batch B    │
└─────────────┘    └─────────────┘
  SYNC GRADIENTS AFTER EACH STEP
```

#### Why This Matters

- Model parallelism (LoRA): Lower memory per GPU, but communication overhead for activations
- Data parallelism (QLoRA): Each GPU needs memory for full model, but only gradient synchronization is needed
- QLoRA's 4-bit quantization makes the full model small enough to fit on each GPU (~0.5 bytes/param vs 2 bytes/param)

### Tradeoffs

QLoRA trades compute for memory:

|Aspect|LoRA|QLoRA|
|---|---|---|
|Memory|Higher (~2 bytes/param)|Lower (~0.5 bytes/param)|
|Training speed|Faster|~20-30% slower due to quantization/dequantization|
|Model quality|Baseline|Slightly lower due to quantization noise|
|Hardware requirement|Higher VRAM|Lower VRAM|

The slower training speed comes from:

- Dequantizing weights before each forward pass
- Quantizing gradients during backward pass
- Double quantization overhead for scaling factors

For most use cases, the memory savings outweigh the speed penalty, especially when training larger models on consumer GPUs.

### Inference

After training, LoRA weights can be:

1. **Merged**: $W' = W + \frac{\alpha}{r} BA$ for zero inference overhead
2. **Kept separate**: Swap adapters for different tasks on the same base model
3. **Quantized inference**: Keep base model quantized with separate LoRA adapters
