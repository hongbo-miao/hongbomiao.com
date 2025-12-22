# Variational Autoencoder (VAE)

## Core Idea

A VAE learns a probabilistic latent representation of data by training an encoder $q_\phi(\mathbf{z}\mid \mathbf{x})$ and decoder $p_\theta(\mathbf{x}\mid \mathbf{z})$ to maximize the Evidence Lower Bound (ELBO).

### Model Architecture

**Encoder** maps input $\mathbf{x}$ to latent distribution parameters:

```math
\mathbf{h} = \mathrm{ReLU}(W_e \mathbf{x} + b_e), \quad
\boldsymbol{\mu} = W_\mu \mathbf{h} + b_\mu, \quad
\log \boldsymbol{\sigma}^2 = W_{\log \sigma^2} \mathbf{h} + b_{\log \sigma^2}.
```

**Reparameterization trick** enables backpropagation through sampling:

```math
\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}).
```

**Decoder** reconstructs from latent code:

```math
\hat{\mathbf{x}} = \sigma(W_d \mathbf{z} + b_d).
```

### Training Objective

The VAE minimizes the negative ELBO:

```math
\mathcal{L}_{\text{VAE}} = \underbrace{\text{BCE}(\hat{\mathbf{x}}, \mathbf{x})}_{\text{reconstruction}} + \underbrace{D_{\mathrm{KL}}\big(q_\phi(\mathbf{z}\mid \mathbf{x}) \,\|\, \mathcal{N}(\mathbf{0}, \mathbf{I})\big)}_{\text{regularization}}.
```

**Reconstruction loss** (binary cross-entropy for MNIST):

```math
\text{BCE} = -\sum_{i=1}^{D} \left[x_i \log \hat{x}_i + (1 - x_i)\log (1 - \hat{x}_i)\right].
```

**Kullback–Leibler (KL) divergence** for diagonal Gaussian posterior:

```math
D_{\mathrm{KL}} = -\tfrac{1}{2} \sum_{j=1}^{J} \left(1 + \log\sigma_j^2 - \mu_j^2 - e^{\log\sigma_j^2}\right).
```

### Training + Outputs

MNIST batches (128 samples) are processed for 10 epochs. Each forward pass:

1. Encodes $\mathbf{x}$ to $(\boldsymbol{\mu}, \log\boldsymbol{\sigma}^2)$
2. Samples $\mathbf{z}$ via reparameterization
3. Decodes to $\hat{\mathbf{x}}$
4. Computes loss and backpropagates

After training:

- `output/vae_reconstruction.png` – original vs reconstructed digits
- `output/vae_generated.png` – samples from $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
