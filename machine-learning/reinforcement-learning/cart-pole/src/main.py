import logging

import gymnasium as gym

logger = logging.getLogger(__name__)


def main() -> None:
    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=42)
    logger.info(observation, info)
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
            logger.info(observation, info)
    env.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
