# AnyUp

## Core Idea

AnyUp is a universal feature upsampling method that takes low-resolution features from any vision encoder (DINOv2, CLIP, ResNet, etc.) and upsamples them to high resolution using guidance from the original high-resolution image. Unlike traditional upsamplers that require encoder-specific training, AnyUp is feature-agnostic and works with any backbone at inference time.

### The Upsampling Problem

Vision Transformers like DINOv2 produce patch-level features at much lower resolution than the input image. For an image $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$ with patch size $P$, the output features have spatial dimensions:

```math
h = \frac{H}{P}, \quad w = \frac{W}{P}.
```

For DINOv2 with $P = 14$ and image size $518 \times 518$, this yields features of size $37 \times 37$, a $14\times$ reduction in spatial resolution. This loss of spatial detail limits performance in dense prediction tasks like segmentation and depth estimation.

### Feature Upsampling with Image Guidance

AnyUp upsamples low-resolution features $\mathbf{F}\_\text{lr} \in \mathbb{R}^{C \times h \times w}$ to high-resolution features $\mathbf{F}\_\text{hr} \in \mathbb{R}^{C \times H \times W}$ using the original image as guidance:

```math
\mathbf{F}_\text{hr} = \text{AnyUp}(\mathbf{I}, \mathbf{F}_\text{lr}).
```

The key insight is that the high-resolution image contains fine-grained spatial information (edges, textures, boundaries) that can guide where and how to interpolate the low-resolution semantic features.

### Attention-Based Upsampling

AnyUp uses a cross-attention mechanism where each high-resolution output position attends to relevant low-resolution feature positions, weighted by image similarity:

```math
\mathbf{F}_\text{hr}(x, y) = \sum_{i,j} \alpha_{(x,y),(i,j)} \cdot \mathbf{F}_\text{lr}(i, j),
```

where:

- $(x, y)$: position in the high-resolution output ($H \times W$ grid)
- $(i, j)$: position in the low-resolution input ($h \times w$ grid)
- $\alpha_{(x,y),(i,j)}$: attention weights computed from image features, ensuring that upsampling respects image boundaries and structure

### Resolution Independence

The upsampler can output features at any target resolution $H' \times W'$:

```math
\mathbf{F}_\text{out} = \text{AnyUp}(\mathbf{I}, \mathbf{F}_\text{lr}, (H', W')) \in \mathbb{R}^{C \times H' \times W'}.
```

By default, it matches the input image resolution $(H, W)$.

### In Code

- Load and normalize image to ImageNet statistics: $\mathbf{I} \in \mathbb{R}^{3 \times H \times W}$ (`load_image`)
- Extract low-resolution features from DINOv2: $\mathbf{F}_\text{lr} \in \mathbb{R}^{C \times h \times w}$ (`extract_dinov2_features`)
- Upsample features using image guidance: $\mathbf{F}_\text{hr} = \text{AnyUp}(\mathbf{I}, \mathbf{F}_\text{lr})$ (`upsample_features_with_anyup`)
- Visualize low-resolution vs high-resolution features: mean over channels for display (`visualize_feature_comparison`)
