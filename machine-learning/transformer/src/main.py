import copy
import logging
import math
import time
from collections import UserDict
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Part 1: Model Architecture
# -----------------------------------------------------------------------------


def create_fixed_positional_encoding(dim: int, max_len: int = 5000) -> Tensor:
    pe = torch.zeros(max_len, dim)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2) * -(math.log(10000.0) / dim),
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0).requires_grad_(requires_grad=False)


class EmbeddingsWithPositionalEncoding(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        dim: int,
        dropout: float = 0.1,
        pe_type: str = "fixed",
        max_len: int = 50000,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocabulary_size, dim)
        self.dim = dim

        if pe_type == "fixed":
            pe = create_fixed_positional_encoding(dim, max_len)
            self.register_buffer("pe", pe)
        else:
            self.pe = nn.Parameter(torch.zeros(1, max_len, dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for embeddings with positional encoding.

        Args:
            x: Tensor, shape (batch_size, seq_len).

        """
        token_embedding = self.embed(x) * math.sqrt(self.dim)
        positional_encoding = self.pe[:, : x.size(1)]
        return self.dropout(token_embedding + positional_encoding)


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, 4 * embed_dim, bias=bias)
        self.linear2 = nn.Linear(4 * embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.linear2(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, (
            "embed_dim (d_model) must be divisible by num_heads"
        )
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_score: Tensor | None = None
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
        return_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        bsz, seq_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert key.size() == value.size()

        q = (
            self.q_proj(query)
            .view(bsz, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(key)
            .view(bsz, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(value)
            .view(bsz, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = (q @ k.transpose(-2, -1)) * self.scaling

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        self.attn_score = attn
        values = attn @ v
        values = values.transpose(1, 2).reshape(bsz, seq_len, embed_dim)
        out = self.out_proj(values)

        if return_weights:
            return out, attn
        return out


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        pre_norm: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, dropout)
        self.norm_self_attn = LayerNorm(embed_dim)
        self.norm_ff = LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        if self.pre_norm:
            norm_x = self.norm_self_attn(x)
            x = x + self.dropout(self.self_attn(norm_x, norm_x, norm_x, mask))
            norm_x = self.norm_ff(x)
            x = x + self.dropout(self.ff(norm_x))
        else:
            x = x + self.dropout(self.self_attn(x, x, x, mask))
            x = self.norm_self_attn(x)
            x = x + self.dropout(self.ff(x))
            x = self.norm_ff(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
        pre_norm: bool = True,
    ) -> None:
        super().__init__()
        encoder_layer = EncoderLayer(embed_dim, num_heads, dropout, pre_norm)
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)],
        )
        self.norm = LayerNorm(embed_dim)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        pre_norm: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, dropout)
        self.norm_self_attn = LayerNorm(embed_dim)
        self.norm_cross_attn = LayerNorm(embed_dim)
        self.norm_ff = LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        source_mask: Tensor | None,
        target_mask: Tensor | None,
    ) -> Tensor:
        if self.pre_norm:
            norm_x = self.norm_self_attn(x)
            x = x + self.dropout(self.self_attn(norm_x, norm_x, norm_x, target_mask))
            norm_x = self.norm_cross_attn(x)
            x = x + self.dropout(self.cross_attn(norm_x, memory, memory, source_mask))
            norm_x = self.norm_ff(x)
            x = x + self.dropout(self.ff(norm_x))
        else:
            x = x + self.dropout(self.self_attn(x, x, x, target_mask))
            x = self.norm_self_attn(x)
            x = x + self.dropout(self.cross_attn(x, memory, memory, source_mask))
            x = self.norm_cross_attn(x)
            x = x + self.dropout(self.ff(x))
            x = self.norm_ff(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
        pre_norm: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(embed_dim, num_heads, dropout, pre_norm)
                for _ in range(num_layers)
            ],
        )
        self.norm = LayerNorm(embed_dim)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        source_mask: Tensor | None,
        target_mask: Tensor | None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


def create_causal_mask(size: int) -> Tensor:
    """Create a causal (lower triangular) mask."""
    attn_shape = (1, size, size)
    causal_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return causal_mask == 0


class Generator(nn.Module):
    def __init__(self, embed_dim: int, vocabulary_size: int) -> None:
        super().__init__()
        self.final_proj = nn.Linear(embed_dim, vocabulary_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return F.log_softmax(self.final_proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(
        self,
        source_vocabulary_size: int,
        target_vocabulary_size: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
        pre_norm: bool = True,
        pe_type: str = "fixed",
    ) -> None:
        super().__init__()
        self.source_embed = EmbeddingsWithPositionalEncoding(
            source_vocabulary_size,
            embed_dim,
            dropout,
            pe_type,
        )
        self.target_embed = EmbeddingsWithPositionalEncoding(
            target_vocabulary_size,
            embed_dim,
            dropout,
            pe_type,
        )
        self.encoder = Encoder(embed_dim, num_layers, num_heads, dropout, pre_norm)
        self.decoder = Decoder(embed_dim, num_layers, num_heads, dropout, pre_norm)
        self.generator = Generator(embed_dim, target_vocabulary_size)
        self.post_init()

    def encode(self, src: Tensor, source_mask: Tensor | None) -> Tensor:
        return self.encoder(self.source_embed(src), source_mask)

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        source_mask: Tensor | None,
        target_mask: Tensor | None,
    ) -> Tensor:
        return self.decoder(self.target_embed(tgt), memory, source_mask, target_mask)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        source_mask: Tensor | None,
        target_mask: Tensor | None,
    ) -> Tensor:
        memory = self.encode(src, source_mask)
        return self.decode(tgt, memory, source_mask, target_mask)

    def post_init(self) -> None:
        self.generator.final_proj.weight = self.target_embed.embed.weight
        logger.info(
            "Source (encoder) and target (decoder) embedding weights are not tied by default.",
        )

    def tie_weights(self) -> None:
        self.source_embed.embed.weight = self.target_embed.embed.weight
        logger.info(
            "Source (encoder) and target (decoder) embedding weights are now tied.",
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


def create_model(
    source_vocabulary_size: int,
    target_vocabulary_size: int,
    embed_dim: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    dropout: float = 0.1,
    pre_norm: bool = True,
    pe_type: str = "fixed",
    device: str | torch.device | None = None,
) -> Transformer:
    """Create and initialize a Transformer model."""
    model = Transformer(
        source_vocabulary_size,
        target_vocabulary_size,
        embed_dim,
        num_layers,
        num_heads,
        dropout,
        pre_norm,
        pe_type,
    )

    # Xavier uniform initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if device is not None:
        model = model.to(device)

    return model


def inference_test() -> None:
    """Test inference with an untrained model."""
    test_model = create_model(
        source_vocabulary_size=11,
        target_vocabulary_size=11,
        embed_dim=512,
        num_layers=2,
        num_heads=8,
        dropout=0.1,
        pre_norm=True,
        pe_type="fixed",
    )
    test_model.generator.final_proj.weight = nn.Parameter(
        torch.randn_like(test_model.generator.final_proj.weight),
    )
    test_model.eval()

    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    source_mask = torch.ones(1, 1, 10)
    memory = test_model.encode(src, source_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for _i in range(9):
        target_mask = create_causal_mask(ys.size(1)).type_as(src.data)
        out = test_model.decode(ys, memory, source_mask, target_mask)
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)],
            dim=1,
        )
    logger.info("Example Untrained Model Prediction: %s", ys)


# -----------------------------------------------------------------------------
# Part 2: Preparation for Training
# -----------------------------------------------------------------------------


class Batch(UserDict):
    """Batch object for training with source, target, and masks."""

    def __init__(self, src: Tensor, tgt: Tensor | None, pad: int) -> None:
        self.data = {}
        self.data["src"] = src
        self.data["source_mask"] = (src != pad).unsqueeze(-2)

        if tgt is not None:
            self.data["tgt"] = tgt[:, :-1]
            self.data["target_y"] = tgt[:, 1:]
            self.data["target_mask"] = self.make_std_mask(self.data["tgt"], pad)
            self.data["num_tokens"] = (self.data["target_y"] != pad).data.sum()

        super().__init__(self.data)

    @staticmethod
    def make_std_mask(tgt: Tensor, pad: int) -> Tensor:
        """Create a mask to hide padding and future words."""
        target_mask = (tgt != pad).unsqueeze(-2)
        return target_mask & create_causal_mask(tgt.size(-1)).type_as(target_mask.data)

    def __getitem__(self, item: str) -> Tensor:
        if isinstance(item, str):
            return self.data[item]
        msg = "Invalid key. Only string keys are available"
        raise KeyError(msg)

    def __getattr__(self, item: str) -> Tensor:
        try:
            return self.data[item]
        except KeyError:
            msg = f"Attribute {item} not found"
            raise AttributeError(msg) from None

    def to(self, device: str | torch.device) -> "Batch":
        """Move batch to specified device."""
        if isinstance(device, str | int | torch.device):
            self.data = {
                k: v.to(device=device) if isinstance(v, torch.Tensor) else v
                for k, v in self.data.items()
            }
        else:
            logger.info(
                "Warning: Attempting to cast a Batch to type %s. This is not supported.",
                device,
            )
        return self


def rate(step: int, model_size: int, factor: float, warmup: int) -> float:
    """Learning rate calculation for original transformer schedule."""
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
) -> float:
    """Lambda function for cosine annealing with warmup."""
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps),
    )
    return max(
        0.0,
        0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
    )


def cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """Create cosine annealing scheduler with warmup."""
    lr_lambda = partial(
        cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# -----------------------------------------------------------------------------
# Part 3: Toy Training Example: Copy input
# -----------------------------------------------------------------------------


def data_generator(
    vocabulary_size: int,
    batch_size: int,
    nbatches: int,
) -> Iterator[Batch]:
    """Generate synthetic data for copy task."""
    for _i in range(nbatches):
        data = torch.randint(1, vocabulary_size, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(requires_grad=False).clone().detach()
        tgt = data.requires_grad_(requires_grad=False).clone().detach()
        yield Batch(src, tgt, pad=0)


@torch.inference_mode()
def greedy_decode(
    model: Transformer,
    src: Tensor,
    source_mask: Tensor,
    max_len: int,
    start_symbol: int,
) -> Tensor:
    """Perform greedy decoding."""
    model.eval()
    memory = model.encode(src, source_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for _i in range(max_len - 1):
        causal_mask = create_causal_mask(ys.size(1)).type_as(src.data)
        out = model.decode(ys, memory, source_mask, causal_mask)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)],
            dim=1,
        )
    return ys


@dataclass
class TrainState:
    """Training state tracker."""

    step: int = 0
    accum_step: int = 0
    samples: int = 0
    num_tokens: int = 0


def run_epoch(
    data_iter: Iterator[Batch],
    model: Transformer,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    mode: str = "train",
    accum_iter: int = 1,
    train_state: TrainState | None = None,
) -> tuple[float, TrainState]:
    """Run one epoch of training or evaluation."""
    if train_state is None:
        train_state = TrainState()

    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0

    for i, batch in enumerate(data_iter):
        batch_on_device = batch.to(model.device)
        logits = model(
            batch_on_device.src,
            batch_on_device.tgt,
            batch_on_device.source_mask,
            batch_on_device.target_mask,
        )
        y_pred = model.generator(logits)
        loss = criterion(
            y_pred.reshape(-1, y_pred.shape[-1]),
            batch_on_device.target_y.reshape(-1),
        )

        if mode in {"train", "train+log"}:
            loss.backward()
            train_state.step += 1
            train_state.samples += batch_on_device.src.shape[0]
            train_state.num_tokens += batch_on_device.num_tokens.item()

            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss.item()
        total_tokens += batch_on_device.num_tokens
        tokens += batch_on_device.num_tokens

        if i % 40 == 1 and (mode in {"train", "train+log"}):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            logger.info(
                "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                "| Tokens / Sec: %7.1f | Learning Rate: %6.1e",
                i,
                n_accum,
                loss,
                tokens / elapsed,
                lr,
            )
            start = time.time()
            tokens = 0

    return total_loss / total_tokens, train_state


def run_toy_example(
    num_epochs: int = 10,
    pre_norm: bool = True,
    device: str = "cpu",
) -> None:
    """Run toy copy task training example."""
    vocabulary_size = 11
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    model = create_model(
        vocabulary_size,
        vocabulary_size,
        embed_dim=512,
        num_layers=2,
        pre_norm=pre_norm,
        device=device,
    )

    num_batches = 20
    batch_size = 80
    warmup_ratio = 0.1
    num_training_steps = num_epochs * num_batches
    num_warmup_steps = math.ceil(num_training_steps * warmup_ratio)
    base_lr = 0.001

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    lr_scheduler = cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps,
        num_training_steps,
    )

    for epoch in range(num_epochs):
        logger.info("\n--- Epoch %d/%d ---", epoch + 1, num_epochs)
        model.train()
        run_epoch(
            data_generator(vocabulary_size, batch_size, nbatches=num_batches),
            model,
            criterion,
            optimizer,
            lr_scheduler,
            mode="train",
        )

    model.eval()
    logger.info("\n--- Inference after training ---")

    # Inference examples
    test_cases = [
        torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
        torch.LongTensor([[0, 1, 3, 5, 7, 9]]),
        torch.LongTensor([[0, 9, 2, 6, 4, 1, 6, 9, 8, 9]]),
    ]

    for i, test_case in enumerate(test_cases, 1):
        source_tensor = test_case.to(model.device)
        max_len = source_tensor.shape[1]
        source_mask = torch.ones(1, 1, max_len).to(model.device)
        pred = greedy_decode(
            model,
            source_tensor,
            source_mask,
            max_len=max_len,
            start_symbol=0,
        )
        logger.info("\nExample %d:", i)
        logger.info("Target:    %s", test_case)
        logger.info("Predicted: %s", pred)


def run_inference_tests() -> None:
    """Run inference tests with untrained model."""
    logger.info("Running inference tests with untrained model...")
    for _ in range(10):
        inference_test()
    logger.info("-" * 50)


def main() -> None:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info("Using device: %s", device)

    # Run inference tests
    run_inference_tests()

    # Run toy examples
    logger.info("--- Running Toy Example (Pre-Norm) ---")
    run_toy_example(num_epochs=20, pre_norm=True, device=device)
    logger.info("\n%s\n", "=" * 50)

    logger.info("--- Running Toy Example (Post-Norm) ---")
    run_toy_example(num_epochs=30, pre_norm=False, device=device)
    logger.info("\n%s\n", "=" * 50)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    main()
