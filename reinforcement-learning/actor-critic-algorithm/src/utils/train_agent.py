import logging
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from services.actor_critic_agent import ActorCriticAgent

logger = logging.getLogger(__name__)


def train_agent(
    environment_name: str = "CartPole-v1",
    episode_count: int = 2000,
    max_step_count: int = 500,
    learning_rate: float = 0.002,
    discount_factor: float = 0.99,
    entropy_coefficient: float = 0.01,
    value_loss_coefficient: float = 0.5,
    hidden_dimension: int = 128,
    reward_threshold: float = 475.0,
    log_interval: int = 100,
    seed: int = 42,
) -> ActorCriticAgent:
    r"""
    Train an Advantage Actor-Critic (A2C) agent.

    The training loop (for each episode):
    1. Collect trajectory: $\{s_t, a_t, r_t\}$ for t = 0, 1, ..., T
    2. Compute returns: $G_t = r_t + \gamma G_{t+1}$
    3. Compute advantages: $A_t = G_t - V(s_t)$
    4. Update actor: $\theta \leftarrow \theta - \alpha \nabla(-\log \pi \cdot A)$
    5. Update critic: $\phi \leftarrow \phi - \alpha \nabla(V - G)^2$

    Key features:
    - Episode-based updates: More stable than step-by-step
    - Advantage normalization: Reduces variance in gradient estimates
    - Entropy bonus: $H(\pi)$ prevents premature convergence
    - Gradient clipping: Prevents exploding gradients

    Args:
        environment_name: Gymnasium environment to train on
        episode_count: Maximum number of training episodes
        max_step_count: Maximum steps per episode $T_{\max}$
        learning_rate: $\alpha$ - learning rate for both actor and critic
        discount_factor: $\gamma$ - for discounting future rewards
        entropy_coefficient: $c_e$ - weight for entropy bonus (exploration)
        value_loss_coefficient: $c_v$ - weight for value function loss
        hidden_dimension: Size of hidden layers in networks
        reward_threshold: Stop training when average reward exceeds this
        log_interval: Print progress every N episodes
        seed: Random seed for reproducibility

    Returns:
        Trained ActorCriticAgent

    """
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    environment = gym.make(environment_name)
    state_dimension = environment.observation_space.shape[0]
    action_dimension = environment.action_space.n

    logger.info(f"Device: {device}")
    logger.info(f"Environment: {environment_name}")
    logger.info(f"State dimension: {state_dimension}")
    logger.info(f"Action dimension: {action_dimension}")

    agent = ActorCriticAgent(
        state_dimension=state_dimension,
        action_dimension=action_dimension,
        device=device,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        entropy_coefficient=entropy_coefficient,
        value_loss_coefficient=value_loss_coefficient,
        hidden_dimension=hidden_dimension,
    )

    recent_rewards: deque[float] = deque(maxlen=100)

    for episode_index in range(episode_count):
        # Seed only first episode for reproducibility; subsequent episodes vary naturally
        reset_seed = seed if episode_index == 0 else None
        state, _ = environment.reset(seed=reset_seed)

        # Episode storage
        states: list[torch.Tensor] = []
        log_probabilities: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        rewards: list[float] = []

        episode_reward = 0.0
        is_terminated = False

        for _ in range(max_step_count):
            state_tensor = torch.FloatTensor(state).to(device)
            states.append(state_tensor)  # $s_t$

            # $a_t \sim \pi(\cdot|s_t)$ - sample action from policy
            action, log_probability, entropy = agent.select_action(state_tensor)
            log_probabilities.append(log_probability)  # $\log \pi(a_t|s_t)$
            entropies.append(entropy)  # $H(\pi(\cdot|s_t))$

            # Environment step: $s_{t+1}, r_t = \text{env}(s_t, a_t)$
            next_state, reward, terminated, truncated, _ = environment.step(action)
            is_done = terminated or truncated
            is_terminated = terminated
            rewards.append(float(reward))  # $r_t$
            episode_reward += reward

            if is_done:
                break

            state = next_state

        # Bootstrap final value for non-terminal states
        # $G_T = 0$ if terminal, $G_T = V(s_T)$ if truncated
        if is_terminated:
            final_value = 0.0
        else:
            with torch.no_grad():
                final_value = agent.critic(
                    torch.FloatTensor(next_state).to(device),
                ).item()

        # $G_t = r_t + \gamma G_{t+1}$ for all t
        returns = agent.compute_returns(
            rewards=rewards,
            is_terminal=is_terminated,
            final_value=final_value,
        )

        # Stack tensors for batch update
        states_tensor = torch.stack(states)
        log_probabilities_tensor = torch.stack(log_probabilities)
        entropies_tensor = torch.stack(entropies)

        # Compute loss and update: $\theta \leftarrow \theta - \alpha \nabla L$
        agent.update(
            states=states_tensor,
            log_probabilities=log_probabilities_tensor,
            entropies=entropies_tensor,
            returns=returns,
        )

        recent_rewards.append(episode_reward)
        average_reward = np.mean(recent_rewards)

        if (episode_index + 1) % log_interval == 0:
            logger.info(
                f"Episode {episode_index + 1}/{episode_count} | "
                f"Reward: {episode_reward:.1f} | "
                f"Average (last 100): {average_reward:.1f}",
            )

        # Early stopping if solved
        if average_reward >= reward_threshold and len(recent_rewards) >= 100:
            logger.info(
                f"Environment solved in {episode_index + 1} episodes! "
                f"Average reward: {average_reward:.1f}",
            )
            break

    environment.close()
    return agent
