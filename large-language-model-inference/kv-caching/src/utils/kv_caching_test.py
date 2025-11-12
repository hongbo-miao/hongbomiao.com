import logging

import torch
from torch import nn
from utils.kv_caching import SimpleSelfAttention

logger = logging.getLogger(__name__)


class TestKvCaching:
    def test_kv_caching(self) -> None:
        torch.manual_seed(42)

        attention = SimpleSelfAttention(
            embed_dim=64,
            num_heads=4,
            max_seq_len=10,
        )
        loss_function = nn.MSELoss()

        x_seq: list[torch.Tensor] = []
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None

        for i in range(3):
            # Simulate a batch of inputs (B, T=1, C)
            # B: Batch size
            # T: Time steps (sequence length)
            # C: channel size (embedding dim)
            x = torch.randn(2, 1, 64, requires_grad=True)
            pos_ids = torch.tensor([[i]])
            y, kv_cache = attention(x, past_kv=kv_cache, pos_ids=pos_ids)

            logger.debug(f"[Step {i + 1}] Output mean: {y.mean().item():.4f}")
            logger.debug(f"[Step {i + 1}] KV Cache size: {kv_cache[0].shape}")

            # Assert that the cache is growing correctly
            assert kv_cache[0].shape[2] == i + 1  # cache length = step count

            # Assert no NaNs
            assert not torch.isnan(y).any()

            x_seq.append(y)

        out = torch.cat(x_seq, dim=1)
        target = torch.zeros_like(out)
        loss = loss_function(out, target)
        loss.backward()

        # Assert loss is a finite scalar
        logger.debug(f"Loss: {loss.item()}")
        assert torch.isfinite(loss)

        # Assert gradients exist for all parameters
        logger.debug(
            f"Grad OK? {all(p.grad is not None for p in attention.parameters())}",
        )
        for p in attention.parameters():
            assert p.grad is not None
