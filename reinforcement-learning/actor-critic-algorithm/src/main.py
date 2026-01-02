import logging

from utils.train_agent import train_agent

logger = logging.getLogger(__name__)


def main() -> None:
    r"""
    Train an Advantage Actor-Critic (A2C) agent on CartPole-v1.

    Actor-Critic Algorithm Overview:
    --------------------------------
    Actor-Critic combines two neural networks working together:

    1. ACTOR (Policy Network) - learns $\pi(a|s)$:
        - Input: State s (cart position, velocity, pole angle, angular velocity)
        - Output: $\pi(a|s)$ = probability distribution over actions (left/right)
        - Loss: $L_{\text{actor}} = -\mathbb{E}[\log \pi(a|s) \cdot A(s,a)]$

    2. CRITIC (Value Network) - learns $V(s)$:
        - Input: State s
        - Output: $V(s) = \mathbb{E}[G_t | s_t = s]$ (expected future reward)
        - Loss: $L_{\text{critic}} = \mathbb{E}[(V(s) - G_t)^2]$

    Key formulas:
    - Return: $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$
    - Advantage: $A_t = G_t - V(s_t)$
    - Policy gradient theorem: $\nabla J = \mathbb{E}[\nabla \log \pi(a|s) \cdot A(s,a)]$
    - Total loss: $L = L_{\text{actor}} + c_v L_{\text{critic}} - c_e H(\pi)$

    Why use advantage instead of raw returns?
    - Raw returns have high variance (absolute value of rewards varies a lot)
    - Advantage centers around zero: positive = better than expected
    - This reduces variance while keeping the gradient unbiased

    Entropy bonus $H(\pi)$:
    - Entropy $H = -\sum \pi \log \pi$ measures randomness in action selection
    - Adding entropy to loss encourages exploration
    - Prevents premature convergence to deterministic (potentially suboptimal) policy
    """
    logger.info("Starting A2C training on CartPole-v1")
    logger.info("Goal: Keep the pole balanced for 475+ steps on average")

    train_agent(
        environment_name="CartPole-v1",
        episode_count=5000,
        max_step_count=500,
        learning_rate=0.002,
        discount_factor=0.99,
        entropy_coefficient=0.01,
        value_loss_coefficient=0.5,
        hidden_dimension=128,
        reward_threshold=475.0,
        log_interval=100,
        seed=42,
    )

    logger.info("Training completed")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
