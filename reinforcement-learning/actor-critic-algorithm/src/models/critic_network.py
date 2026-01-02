import torch
from torch import nn


class CriticNetwork(nn.Module):
    r"""
    Critic network that estimates the state value function V(s).

    The critic evaluates how good a state is by predicting the expected
    cumulative reward from that state. It provides the baseline for
    reducing variance in the policy gradient.

    Math:
        Value function:
        $$
        V(s) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s\right]
        $$
        This is the expected discounted sum of future rewards starting from state s.

    Architecture:
        Input: state vector
        Hidden layers: fully connected with Tanh activation
        Output: single scalar value estimate $V(s)$
    """

    def __init__(
        self,
        state_dimension: int,
        hidden_dimension: int = 128,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dimension, hidden_dimension),
            nn.Tanh(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.Tanh(),
            nn.Linear(hidden_dimension, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        r"""
        Estimate the value of a state.

        $V(s) = \mathbb{E}[G_t \mid s_t = s]$ where $G_t$ is the return from time t.
        """
        return self.network(state)
