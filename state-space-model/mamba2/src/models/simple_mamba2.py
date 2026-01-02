import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class SimpleMamba2(nn.Module):
    r"""
    A simplified Mamba 2 block.

    Mamba 2 uses the State Space Duality (SSD) framework which shows:
    - SSM with scalar $A$ is equivalent to linear attention
    - $y = (L \odot (C B^T)) X$, where $L$ is a causal decay mask

    Args:
        dimension: Input/output dimension (d_model)
        state_dimension: SSM state dimension (d_state), also called head_dim
        head_count: Number of heads
        expand_factor: Expansion factor for inner dimension

    """

    def __init__(
        self,
        dimension: int,
        state_dimension: int = 64,
        head_count: int = 8,
        expand_factor: int = 2,
    ) -> None:
        super().__init__()

        self.dimension = dimension
        self.state_dimension = state_dimension
        self.head_count = head_count
        self.inner_dimension = dimension * expand_factor
        assert self.inner_dimension % self.head_count == 0, (
            f"inner_dimension ({self.inner_dimension}) must be divisible by "
            f"head_count ({self.head_count})"
        )
        self.head_dimension = self.inner_dimension // head_count

        # Input projection for X (goes through conv -> SSM)
        self.x_projection = nn.Linear(dimension, self.inner_dimension, bias=False)

        # Input projection for Z (gate path)
        self.z_projection = nn.Linear(dimension, self.inner_dimension, bias=False)

        # Causal 1D convolution for local context
        self.causal_convolution = nn.Conv1d(
            in_channels=self.inner_dimension,
            out_channels=self.inner_dimension,
            kernel_size=4,
            padding=3,
            groups=self.inner_dimension,
        )

        # SSM parameter projections (input-dependent)
        # B and C projections: per head
        self.b_projection = nn.Linear(
            self.inner_dimension,
            head_count * state_dimension,
            bias=False,
        )
        self.c_projection = nn.Linear(
            self.inner_dimension,
            head_count * state_dimension,
            bias=False,
        )

        # Delta (dt) projection: controls selectivity
        self.dt_projection = nn.Linear(self.inner_dimension, head_count, bias=True)

        # A parameter: scalar per head (log-space for stability)
        # In Mamba 2, A is simplified to scalar (not diagonal)
        self.log_a = nn.Parameter(torch.log(torch.ones(head_count) * 0.5))

        # D parameter: skip connection
        self.d_parameter = nn.Parameter(torch.ones(head_count))

        # Output normalization (group norm per head)
        self.output_norm = nn.GroupNorm(head_count, self.inner_dimension)

        # Output projection
        self.output_projection = nn.Linear(self.inner_dimension, dimension, bias=False)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass through the Mamba 2 block.

        The key insight of Mamba 2 (SSD):
        SSM can be computed as $y = (L \odot (C B^T)) X$
        where $L[i,j] = \exp(\sum_{k=j}^{i} \log a_k)$ for $i \geq j$

        Args:
            input_tensor: Shape (batch_size, sequence_length, dimension)

        Returns:
            Output tensor of shape (batch_size, sequence_length, dimension)

        """
        batch_size, sequence_length, _ = input_tensor.shape

        # Step 1: Input projections
        x = self.x_projection(input_tensor)  # (batch, seq, inner_dim)
        z = self.z_projection(input_tensor)  # (batch, seq, inner_dim)

        # Step 2: Causal convolution on x path
        x = x.transpose(1, 2)  # (batch, inner_dim, seq)
        x = self.causal_convolution(x)[:, :, :sequence_length]
        x = x.transpose(1, 2)  # (batch, seq, inner_dim)
        x = F.silu(x)

        # Step 3: Compute input-dependent SSM parameters
        # B: (batch, seq, head_count * state_dim)
        b = self.b_projection(x)
        b = b.view(batch_size, sequence_length, self.head_count, self.state_dimension)

        # C: (batch, seq, head_count * state_dim)
        c = self.c_projection(x)
        c = c.view(batch_size, sequence_length, self.head_count, self.state_dimension)

        # Delta (dt): (batch, seq, head_count)
        dt = F.softplus(self.dt_projection(x))

        # Reshape x for multi-head: (batch, seq, head_count, head_dim)
        x = x.view(batch_size, sequence_length, self.head_count, self.head_dimension)

        # Step 4: Compute SSM output using SSD
        # A is scalar per head
        a = -torch.exp(self.log_a)  # (head_count,)

        y = self._ssd_compute(
            x=x,
            b=b,
            c=c,
            dt=dt,
            a=a,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )

        # Add skip connection with D
        # y: (batch, seq, head_count, head_dim)
        # d: (head_count,) -> (1, 1, head_count, 1)
        y = y + self.d_parameter.view(1, 1, -1, 1) * x

        # Reshape back: (batch, seq, inner_dim)
        y = y.reshape(batch_size, sequence_length, self.inner_dimension)

        # Step 5: Output normalization
        y = self.output_norm(y.transpose(1, 2)).transpose(1, 2)

        # Step 6: Gating with z
        output = y * F.silu(z)

        # Step 7: Output projection
        return self.output_projection(output)

    def _ssd_compute(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        dt: torch.Tensor,
        a: torch.Tensor,
        batch_size: int,
        sequence_length: int,
    ) -> torch.Tensor:
        r"""
        Compute SSM using State Space Duality.

        The SSD insight: SSM is equivalent to:

        $$
        y_t = \sum_{s \leq t} (\text{decay from } s \text{ to } t) \cdot (C_t \cdot B_s) \cdot x_s
        $$

        This is like attention: $y = \text{softmax}(QK^T) V$
        But with: $y = (L \odot (C B^T)) X$
        where $L$ is the causal decay mask

        Args:
            x: Input, shape (batch, seq, head_count, head_dim)
            b: B parameter, shape (batch, seq, head_count, state_dim)
            c: C parameter, shape (batch, seq, head_count, state_dim)
            dt: Delta, shape (batch, seq, head_count)
            a: A parameter (scalar per head), shape (head_count,)
            batch_size: Batch size
            sequence_length: Sequence length

        Returns:
            SSM output, shape (batch, seq, head_count, head_dim)

        """
        # Initialize state: (batch, head_count, state_dim, head_dim)
        state = torch.zeros(
            batch_size,
            self.head_count,
            self.state_dimension,
            self.head_dimension,
            device=x.device,
            dtype=x.dtype,
        )

        outputs = []

        for t in range(sequence_length):
            # Get values at time t
            x_t = x[:, t, :, :]  # (batch, head_count, head_dim)
            b_t = b[:, t, :, :]  # (batch, head_count, state_dim)
            c_t = c[:, t, :, :]  # (batch, head_count, state_dim)
            dt_t = dt[:, t, :]  # (batch, head_count)

            # Discretize A: $\text{decay} = \exp(\Delta t \cdot a)$
            # dt_t: (batch, head_count), a: (head_count,)
            # decay: (batch, head_count, 1, 1) for broadcasting
            decay = torch.exp(dt_t * a).unsqueeze(-1).unsqueeze(-1)

            # Discretize B: $\bar{B} = \Delta t \cdot B$
            # dt_t: (batch, head_count) -> (batch, head_count, 1)
            # b_t: (batch, head_count, state_dim)
            b_bar = dt_t.unsqueeze(-1) * b_t  # (batch, head_count, state_dim)

            # State update: $h = \text{decay} \cdot h + \bar{B} \otimes x$
            # b_bar: (batch, head_count, state_dim) -> (batch, head_count, state_dim, 1)
            # x_t: (batch, head_count, head_dim) -> (batch, head_count, 1, head_dim)
            # outer product: (batch, head_count, state_dim, head_dim)
            state = decay * state + b_bar.unsqueeze(-1) * x_t.unsqueeze(-2)

            # Output: $y = C \cdot h$ (contract over state_dim)
            # c_t: (batch, head_count, state_dim) -> (batch, head_count, state_dim, 1)
            # state: (batch, head_count, state_dim, head_dim)
            # sum over state_dim: (batch, head_count, head_dim)
            y_t = (c_t.unsqueeze(-1) * state).sum(dim=-2)

            outputs.append(y_t)

        # Stack: (batch, seq, head_count, head_dim)
        return torch.stack(outputs, dim=1)
