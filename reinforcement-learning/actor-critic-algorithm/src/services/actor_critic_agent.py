import torch
from models.actor_network import ActorNetwork
from models.critic_network import CriticNetwork
from torch import optim


class ActorCriticAgent:
    r"""
    Advantage Actor-Critic (A2C) agent with batched episode updates.

    The actor learns the policy: which action to take in each state.
    The critic learns the value function: how good each state is.

    This implementation collects full episodes before updating,
    which provides more stable gradient estimates.

    Math:
        Return: $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$
        Advantage: $A_t = G_t - V(s_t)$
        Policy gradient: $\nabla_\theta J = \mathbb{E}[\nabla \log \pi(a|s) \cdot A(s,a)]$
        Actor loss: $L_{\text{actor}} = -\mathbb{E}[\log \pi(a|s) \cdot A]$
        Critic loss: $L_{\text{critic}} = \mathbb{E}[(V(s) - G)^2]$
        Entropy bonus: $H(\pi) = -\sum \pi \log \pi$
        Total loss: $L = L_{\text{actor}} + c_v L_{\text{critic}} - c_e H$
    """

    def __init__(
        self,
        state_dimension: int,
        action_dimension: int,
        device: torch.device,
        learning_rate: float = 0.002,
        discount_factor: float = 0.99,
        entropy_coefficient: float = 0.01,
        value_loss_coefficient: float = 0.5,
        hidden_dimension: int = 128,
    ) -> None:
        self.device = device
        self.discount_factor = discount_factor
        self.entropy_coefficient = entropy_coefficient
        self.value_loss_coefficient = value_loss_coefficient

        self.actor = ActorNetwork(
            state_dimension=state_dimension,
            action_dimension=action_dimension,
            hidden_dimension=hidden_dimension,
        ).to(device)
        self.critic = CriticNetwork(
            state_dimension=state_dimension,
            hidden_dimension=hidden_dimension,
        ).to(device)

        # Single optimizer for both networks (common in A2C)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate,
        )

    def select_action(
        self,
        state: torch.Tensor,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        return self.actor.select_action(state)

    def compute_returns(
        self,
        rewards: list[float],
        is_terminal: bool,
        final_value: float,
    ) -> torch.Tensor:
        r"""
        Compute discounted returns for each timestep.

        Math:
            Recursive: $G_t = r_t + \gamma G_{t+1}$
            Expanded: $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$

        If episode didn't terminate, bootstrap from the final value estimate:
            $G_T = V(s_T)$ instead of 0
        """
        returns = []
        # Bootstrap: $G_T = 0$ if terminal, $G_T = V(s_T)$ if truncated
        running_return = 0.0 if is_terminal else final_value

        # Compute returns backwards: $G_t = r_t + \gamma G_{t+1}$
        for reward in reversed(rewards):
            running_return = reward + self.discount_factor * running_return
            returns.insert(0, running_return)

        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    def update(
        self,
        states: torch.Tensor,
        log_probabilities: torch.Tensor,
        entropies: torch.Tensor,
        returns: torch.Tensor,
    ) -> tuple[float, float, float]:
        r"""
        Update actor and critic using collected episode data.

        Math:
        $$
        L_{\text{total}} = L_{\text{actor}} + c_v L_{\text{critic}} - c_e H
        $$

        Where:
            $L_{\text{actor}} = -\mathbb{E}[\log \pi(a|s) \cdot A]$ (policy gradient with advantage)
            $L_{\text{critic}} = \mathbb{E}[(V(s) - G)^2]$ (value function MSE)
            $H = \mathbb{E}[-\sum \pi \log \pi]$ (entropy bonus for exploration)

        Returns:
            Tuple of (policy_loss, value_loss, entropy)

        """
        # $V(s_t)$ - critic's estimate of state values
        values = self.critic(states).squeeze()

        # $A_t = G_t - V(s_t)$ - advantage: how much better than expected?
        # detach() prevents gradient flow to critic through advantage
        advantages = returns - values.detach()

        # Normalize advantages for training stability (zero mean, unit variance)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # $L_{\text{actor}} = -\mathbb{E}[\log \pi(a|s) \cdot A]$
        # Negative because we minimize loss but want to maximize expected return
        policy_loss = -(log_probabilities * advantages).mean()

        # $L_{\text{critic}} = \mathbb{E}[(V(s) - G)^2]$ - MSE between predicted and actual returns
        value_loss = torch.nn.functional.mse_loss(values, returns)

        # $H(\pi) = \mathbb{E}[\text{entropy}]$ - encourages exploration
        entropy = entropies.mean()

        # $L = L_{\text{actor}} + c_v L_{\text{critic}} - c_e H$
        # Subtract entropy because higher entropy should lower total loss
        total_loss = (
            policy_loss
            + self.value_loss_coefficient * value_loss
            - self.entropy_coefficient * entropy
        )

        # Gradient descent: $\theta \leftarrow \theta - \alpha \nabla L$
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            max_norm=0.5,
        )
        self.optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy.item()
