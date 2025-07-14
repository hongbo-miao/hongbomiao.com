import logging

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

logger = logging.getLogger(__name__)


class SimpleSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int = 128,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # (max_seq_len, embed_dim)
        self.pos_embedding = nn.Parameter(
            torch.randn(max_seq_len, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,  # (B, T=1, C)
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        pos_ids: torch.Tensor | None = None,  # (B, T)
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # Batch size, time steps (sequence length), and channel size (embedding dim)
        B, T, C = x.size()  # noqa: N806
        if pos_ids is None:
            # `.arange(T, device=x.device)` creates a tensor of shape (T,) with values from 0 to T-1.
            # `.unsqueeze(0)` adds a new dimension at the front, making it (1, T).
            pos_ids = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T=1)

        pos_embedding = self.pos_embedding[pos_ids]  # (B, T=1, C)
        x = x + pos_embedding  # (B, T=1, C)

        # This line performs a linear transformation on the last dimension (C).
        # The shape doesn't change, but the values inside the tensor are now the Query representations, not the original embeddings.
        q = self.q_proj(x)  # (B, T_q=1, C)
        k = self.k_proj(x)  # (B, T_k=T+1, C)
        v = self.v_proj(x)  # (B, T_v=1, C)

        # `.view(B, T, self.num_heads, self.head_dim)`
        # (B, T_q=1, C) -> (B, T_q=1, num_heads, head_dim)
        # It takes the last dimension (C = 64) and splits it into two new dimensions: num_heads (4) and head_dim (16).
        # This is the "Multi-Head" part of multi-head attention. Instead of having one attention mechanism look at the full 64-dimension embedding, we create 4 parallel "heads" that each look at a smaller 16-dimension slice. This allows the model to focus on different kinds of relationships simultaneously.

        # `.transpose(1, 2)`
        # (B, T_q=1, num_heads, head_dim) -> (B, num_heads, T_q=1, head_dim)
        # PyTorch expect the tensor to be in the format (batch, heads, seq_len, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            # Append new keys/values to cache
            # Concatenate (join) two tensors along dimension 2
            k = torch.cat([past_k, k], dim=2)  # (B, num_heads, T_k=T+1, head_dim)
            v = torch.cat([past_v, v], dim=2)

        # Save cache
        present_kv = (k, v)

        # q: (B, num_heads, T_q=1, head_dim)
        # k: (B, num_heads, T_k=T+1, head_dim)
        # q @ k.transpose(-2, -1): (B, num_heads, T_q=1, T_k=T+1)
        #
        # `/ (self.head_dim**0.5)`
        # This is the "Scaled" part of Scaled Dot-Product Attention.
        # It scales the scores down to prevent them from becoming too large, which helps stabilize training.
        # It does not change the tensor's shape, which remains (B, num_heads, T_q=1, T_k=T+1).
        attention_scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)

        # The raw scores are just numbers.
        # softmax converts these scores into a probability distribution (i.e., weights that sum to 1).
        # Applying it on the last dimension (dim=-1) means that for our single query token, the attention it pays across all T_k=T+1 key tokens will sum to 1.
        # The shape of attention_weights remains (B, num_heads, T_q=1, T_k=T+1).
        attention_weights = F.softmax(attention_scores, dim=-1)

        # (B, num_heads, T_q=1, head_dim) = (B, num_heads, T_q=1, T_k=T+1) @ (B, num_heads, T_k=T+1, head_dim)
        attention_output = attention_weights @ v

        # reshape back into the standard: (B, T, C)
        attention_output = attention_output.transpose(1, 2).contiguous().view(B, T, C)

        # The final reshaped output is passed through one last linear layer `self.out_proj`.
        # This allows the model to learn how to best mix the information gathered from the different attention heads.
        # The shape remains (B, T, C).
        return self.out_proj(attention_output), present_kv
