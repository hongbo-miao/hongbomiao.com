import torch
from torch import nn
from torch.distributions import Categorical


class ActorNetwork(nn.Module):
    r"""
    Actor network that learns a policy mapping states to action probabilities.

    The actor outputs a probability distribution over discrete actions.
    It is trained to maximize expected returns using policy gradients.

    Math:
        Policy: $\pi(a|s) = P(\text{action} = a | \text{state} = s)$
        Output:
        $$
        \pi(a|s) = \text{softmax}(z) = \frac{e^{z_a}}{\sum_{a'} e^{z_{a'}}}
        $$

    Architecture:
        Input: state vector (e.g., cart position, velocity, pole angle, angular velocity)
        Hidden layers: fully connected with Tanh activation
        Output: action probabilities via softmax
    """

    def __init__(
        self,
        state_dimension: int,
        action_dimension: int,
        hidden_dimension: int = 128,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dimension, hidden_dimension),
            nn.Tanh(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.Tanh(),
            nn.Linear(hidden_dimension, action_dimension),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        r"""
        Compute action logits for a given state.

        $$
        \pi(a|s) = \text{softmax}(z) = \frac{e^{z_a}}{\sum_{a'} e^{z_{a'}}}
        $$

        Returns raw logits (before softmax). When passing logits to Categorical(logits=...),
        it handles the softmax internally via log-softmax, which is more numerically stable.
        """
        return self.network(state)

    def select_action(
        self,
        state: torch.Tensor,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        r"""
        Select an action based on the current policy.

        Math:
            Action sampling: $a \sim \pi(\cdot|s)$
            Log probability: $\log \pi(a|s)$ - used in policy gradient
            Entropy: $H(\pi) = -\sum_a \pi(a|s) \log \pi(a|s)$ - measures exploration

        Returns:
            A tuple of (selected_action, log_probability, entropy)

        The entropy measures how "spread out" the action distribution is.
        High entropy = more exploration, low entropy = more exploitation.

        """
        logits = self.forward(state)
        distribution = Categorical(logits=logits)
        action = distribution.sample()  # $a \sim \pi(\cdot|s)$
        log_probability = distribution.log_prob(action)  # $\log \pi(a|s)$
        entropy = distribution.entropy()  # $H(\pi) = -\sum p \log p$
        return action.item(), log_probability, entropy
