import logging
from dataclasses import asdict
from pprint import pformat

import torch
from opentau.configs import parser
from opentau.configs.train import TrainPipelineConfig
from opentau.policies.factory import get_policy_class
from opentau.utils.random_utils import set_seed
from opentau.utils.utils import (
    attempt_torch_compile,
    auto_torch_device,
    create_dummy_observation,
    init_logging,
)

logger = logging.getLogger(__name__)


@parser.wrap()
def main(config: TrainPipelineConfig) -> None:
    logger.info(pformat(asdict(config)))

    device = auto_torch_device()

    if config.seed is not None:
        set_seed(config.seed)

    logger.info("Creating policy")
    policy_class = get_policy_class(config.policy.type)
    policy = policy_class.from_pretrained(
        config.policy.pretrained_path,
        config=config.policy,
    )
    policy.to(device=device, dtype=torch.bfloat16)
    policy.eval()
    policy = attempt_torch_compile(policy, device_hint=device)

    policy.reset()

    observation = create_dummy_observation(config, device, dtype=torch.bfloat16)

    logger.info(f"Observation keys: {observation.keys()}")

    with torch.inference_mode():
        for _ in range(1000):
            action = policy.select_action(observation)
            action = action.to("cpu", torch.float32).numpy()
            logger.info(f"Output shape: {action.shape}")

    logger.info("End of inference")


if __name__ == "__main__":
    init_logging()
    main()
