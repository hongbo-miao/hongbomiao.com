# DINOv3

## Core Idea

DINOv3 (self-DIstillation with NO labels) is a self-supervised Vision Transformer that learns patch-level embeddings useful for semantic correspondence. This project uses DINOv3 to compute **cosine similarity** between patches for cross-image semantic matching.

### Image Padding

ViT requires image dimensions to be divisible by patch size $P$. Given an input image $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$, pad dimensions to the nearest multiple of $P$:

```math
H' = \left\lceil \frac{H}{P} \right\rceil \cdot P, \quad W' = \left\lceil \frac{W}{P} \right\rceil \cdot P.
```

The padded image $\mathbf{I}' \in \mathbb{R}^{H' \times W' \times 3}$ is created by placing the original image at the top-left and filling the right/bottom margins with zeros (black).

### Image to Patches

The padded image $\mathbf{I}'$ is divided into non-overlapping patches of size $P \times P$:

```math
N_\text{row} = \frac{H'}{P}, \quad N_\text{col} = \frac{W'}{P}, \quad N_\text{patches} = N_\text{row} \times N_\text{col}.
```

Since $H'$ and $W'$ are exact multiples of $P$, these divisions yield integers.

Each patch is linearly embedded into a $D$-dimensional vector. The ViT outputs:

```math
\mathbf{H} = [\mathbf{h}_\text{CLS}, \mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_{N_\text{patches}}] \in \mathbb{R}^{(1 + N_\text{patches}) \times D}.
```

where:

- $\mathbf{h}_\text{CLS}$ (index 0): A learnable CLS (classification) token prepended to the patch sequence. It aggregates global image information and is typically used for image-level tasks like classification.
- $\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_{N_\text{patches}}$ (indices 1 to $N_\text{patches}$): Patch embeddings that retain spatial locality and encode local visual features.

This project uses the patch embeddings $\mathbf{h}_i$ (excluding the CLS token) because we need **spatially localized** features for pixel-level similarity matching, not a global image representation.

### Pixel to Patch Index Mapping

Given a pixel coordinate $(x, y)$, the corresponding patch index is:

```math
\text{col} = \left\lfloor \frac{x}{P} \right\rfloor, \quad \text{row} = \left\lfloor \frac{y}{P} \right\rfloor, \quad \text{index} = \text{row} \times N_\text{col} + \text{col}.
```

### Embedding Normalization

Each patch embedding is L2-normalized:

```math
\hat{\mathbf{h}}_i = \frac{\mathbf{h}_i}{\lVert \mathbf{h}_i \rVert_2 + \epsilon}, \quad \epsilon = 10^{-8}.
```

### Cosine Similarity

Given a query patch embedding $\hat{\mathbf{q}}$ (the normalized embedding at the clicked location), the cosine similarity with each patch $i$ is:

```math
s_i = \hat{\mathbf{h}}_i \cdot \hat{\mathbf{q}} = \sum_{d=1}^{D} \hat{h}_{i,d} \cdot \hat{q}_d.
```

Since both vectors are normalized, this equals:

```math
s_i = \cos(\theta_i) = \frac{\mathbf{h}_i \cdot \mathbf{q}}{\lVert \mathbf{h}_i \rVert_2 \cdot \lVert \mathbf{q} \rVert_2}.
```

### Similarity Map Visualization

The similarity scores are reshaped into a 2D map and min-max normalized for display:

```math
\mathbf{S} = [s_1, s_2, \dots, s_{N_\text{patches}}] \in \mathbb{R}^{N_\text{row} \times N_\text{col}}.
```

```math
\tilde{s}_i = \frac{s_i - \min(\mathbf{S})}{\max(\mathbf{S}) - \min(\mathbf{S}) + \epsilon}.
```

The normalized map $\tilde{\mathbf{S}} \in [0, 1]$ is upsampled to the original image resolution using nearest-neighbor interpolation and overlaid as a heatmap.

### In Code

- Load image and pad to multiple of patch size $P$: $H' = \lceil H/P \rceil \cdot P$ (`pad_and_normalize_image`)
- Extract patch embeddings from ViT: $\mathbf{h}_i \in \mathbb{R}^D$ (`create_patch_image_state`)
- Normalize embeddings: $\hat{\mathbf{h}}_i = \mathbf{h}_i / (\lVert \mathbf{h}_i \rVert_2 + \epsilon)$ (`embeddings_normalized`)
- Convert click $(x, y)$ to patch index: $\text{index} = \lfloor y/P \rfloor \cdot N_\text{col} + \lfloor x/P \rfloor$ (`convert_pixel_coordinates_to_patch_index`)
- Compute cosine similarity via matrix multiplication: $s_i = \hat{\mathbf{h}}_i \cdot \hat{\mathbf{q}}$ (`create_similarity_images`)
- Min-max normalize and visualize: $\tilde{s}_i = (s_i - \min(\mathbf{S})) / (\max(\mathbf{S}) - \min(\mathbf{S}) + \epsilon)$ (`create_similarity_images`)
