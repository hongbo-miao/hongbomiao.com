import logging

import numpy as np
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config

logger = logging.getLogger(__name__)

MODEL_NAME = "pi05_droid"
CHECKPOINT_URL = f"gs://openpi-assets/checkpoints/{MODEL_NAME}"


def create_dummy_droid_observation() -> dict:
    random_number_generator = np.random.default_rng()
    return {
        "observation/exterior_image_1_left": random_number_generator.integers(
            256,
            size=(224, 224, 3),
            dtype=np.uint8,
        ),
        "observation/wrist_image_left": random_number_generator.integers(
            256,
            size=(224, 224, 3),
            dtype=np.uint8,
        ),
        "observation/joint_position": random_number_generator.random(7),
        "observation/gripper_position": random_number_generator.random(1),
        "prompt": "pick up the object",
    }


def main() -> None:
    logger.info(f"Loading {MODEL_NAME} model configuration")
    model_config = config.get_config(MODEL_NAME)

    logger.info("Downloading model checkpoint")
    checkpoint_directory = download.maybe_download(CHECKPOINT_URL)

    logger.info("Creating trained policy from checkpoint")
    policy = policy_config.create_trained_policy(model_config, checkpoint_directory)

    logger.info("Running inference on dummy observation")
    observation = create_dummy_droid_observation()
    result = policy.infer(observation)

    action_chunk = result["actions"]
    logger.info(f"Predicted actions shape: {action_chunk.shape}")
    logger.info(f"Predicted actions: {action_chunk}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        # force=True ensures logging config applies even if openpi has already configured logging during import
        force=True,
    )

    main()
