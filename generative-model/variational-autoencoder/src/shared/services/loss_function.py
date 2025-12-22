import torch
import torch.nn.functional as F  # noqa: N812


def loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy + KL divergence, as described in README's training objective."""
    bce = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld
