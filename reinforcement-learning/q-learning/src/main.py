import logging
from random import SystemRandom

import numpy as np

logger = logging.getLogger(__name__)


class TrainingEnvironment:
    """
    Discrete environment.

    - States: 0, 1, 2, 3
    - Actions: 0 (left), 1 (right)
    - Goal: reach state 3.
    """

    def __init__(self) -> None:
        self.state: int = 0
        self.n_states: int = 4
        self.n_actions: int = 2

    def reset(self) -> int:
        self.state = 0
        return self.state

    def step(self, action: int) -> tuple[int, float, bool]:
        if action == 1:
            self.state = min(self.state + 1, self.n_states - 1)
        else:
            self.state = max(self.state - 1, 0)
        reward: float = 1.0 if self.state == self.n_states - 1 else 0.0
        is_done: bool = self.state == self.n_states - 1
        return self.state, reward, is_done


def train_q_table(
    training_environment: TrainingEnvironment,
    alpha: float,
    gamma: float,
    epsilon: float,
    episode_number: int,
) -> np.ndarray:
    """
    Train Q-learning algorithm.

    Args:
        training_environment: Environment instance.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Exploration rate.
        episode_number: Number of training episodes.

    Returns:
        Trained Q-table as a NumPy array.

    """
    q_table: np.ndarray = np.zeros(
        (training_environment.n_states, training_environment.n_actions),
    )
    random_number_generator = SystemRandom()
    for _ in range(episode_number):
        state: int = training_environment.reset()
        is_done: bool = False
        while not is_done:
            # ε-greedy policy
            if random_number_generator.random() < epsilon:
                action: int = random_number_generator.randint(
                    0,
                    training_environment.n_actions - 1,
                )
            else:
                action = int(np.argmax(q_table[state]))
            next_state, reward, is_done = training_environment.step(action)
            # Q-learning update: $Q(s,a) \leftarrow Q(s,a) + \alpha \cdot (TD_{target} - Q(s,a))$
            # Temporal difference (TD) target: $TD_{target} = r + \gamma \cdot \max_{a'} Q(s', a')$
            # Temporal difference (TD) error: $TD_{error} = TD_{target} - Q(s, a)$
            next_action: int = int(np.argmax(q_table[next_state]))
            q_table[state, action] += alpha * (
                reward
                + gamma * q_table[next_state, next_action]
                - q_table[state, action]
            )
            state = next_state
    return q_table


def evaluate_policy(
    training_environment: TrainingEnvironment,
    q_table: np.ndarray,
) -> list[int]:
    """
    Evaluate the learned policy and return the trajectory.

    Args:
        training_environment: Environment instance.
        q_table: Learned Q-table.

    Returns:
        List of visited states following the greedy policy.

    """
    state: int = training_environment.reset()
    is_done: bool = False
    trajectory: list[int] = [state]
    while not is_done:
        action: int = int(np.argmax(q_table[state]))
        next_state, _, is_done = training_environment.step(action)
        trajectory.append(next_state)
        state = next_state
    return trajectory


def main() -> None:
    training_environment = TrainingEnvironment()
    q_table = train_q_table(
        training_environment,
        alpha=0.8,
        gamma=0.95,
        epsilon=0.1,
        episode_number=500,
    )
    logger.info(f"Training finished. Learned Q-table:\n{q_table}")
    trajectory = evaluate_policy(training_environment, q_table)
    logger.info(f"Optimal trajectory: {trajectory}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
