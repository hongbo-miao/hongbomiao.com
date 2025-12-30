import logging
from random import SystemRandom

import torch
from PIL import Image
from shared.image.utils.create_altitude_image import create_altitude_image
from torch import Tensor
from vision_language_action_lib.types.action_output import ActionOutput

secure_random = SystemRandom()

logger = logging.getLogger(__name__)


LANDING_INSTRUCTIONS = [
    "landing",
    "land",
    "descend",
    "go down",
    "land on the ground",
    "descend to ground",
    "lower altitude",
]

TAKEOFF_INSTRUCTIONS = [
    "take off",
    "takeoff",
    "go up",
    "ascend",
    "rise",
    "climb",
    "increase altitude",
    "fly up",
]


def generate_flight_demonstrations(
    sample_count: int,
    image_size: tuple[int, int] = (256, 256),
) -> list[dict[str, Image.Image | str | ActionOutput]]:
    demonstrations = []

    landing_count = sample_count // 2
    takeoff_count = sample_count - landing_count

    for _ in range(landing_count):
        demo = generate_landing_demonstration(image_size=image_size)
        demonstrations.append(demo)

    for _ in range(takeoff_count):
        demo = generate_takeoff_demonstration(image_size=image_size)
        demonstrations.append(demo)

    secure_random.shuffle(demonstrations)

    logger.info(
        f"Generated {sample_count} flight demonstrations ({landing_count} landing, {takeoff_count} takeoff)",
    )
    return demonstrations


def generate_landing_demonstration(
    image_size: tuple[int, int] = (256, 256),
) -> dict[str, Image.Image | str | ActionOutput]:
    simulated_altitude = secure_random.uniform(0.5, 10.0)

    if simulated_altitude > 5.0:
        descent_rate = secure_random.uniform(-0.8, -0.4)
    elif simulated_altitude > 2.0:
        descent_rate = secure_random.uniform(-0.4, -0.2)
    else:
        descent_rate = secure_random.uniform(-0.2, -0.05)

    image = create_altitude_image(
        simulated_altitude=simulated_altitude,
        image_size=image_size,
    )

    action = ActionOutput(
        delta_x=secure_random.uniform(-0.05, 0.05),
        delta_y=secure_random.uniform(-0.05, 0.05),
        delta_z=descent_rate,
        delta_roll=secure_random.uniform(-0.02, 0.02),
        delta_pitch=secure_random.uniform(-0.02, 0.02),
        delta_yaw=secure_random.uniform(-0.02, 0.02),
    )

    return {
        "image": image,
        "instruction": secure_random.choice(LANDING_INSTRUCTIONS),
        "action": action,
    }


def generate_takeoff_demonstration(
    image_size: tuple[int, int] = (256, 256),
) -> dict[str, Image.Image | str | ActionOutput]:
    simulated_altitude = secure_random.uniform(0.0, 5.0)

    if simulated_altitude < 1.0:
        ascent_rate = secure_random.uniform(0.4, 0.8)
    elif simulated_altitude < 3.0:
        ascent_rate = secure_random.uniform(0.2, 0.5)
    else:
        ascent_rate = secure_random.uniform(0.1, 0.3)

    image = create_altitude_image(
        simulated_altitude=simulated_altitude,
        image_size=image_size,
    )

    action = ActionOutput(
        delta_x=secure_random.uniform(-0.05, 0.05),
        delta_y=secure_random.uniform(-0.05, 0.05),
        delta_z=ascent_rate,
        delta_roll=secure_random.uniform(-0.02, 0.02),
        delta_pitch=secure_random.uniform(-0.02, 0.02),
        delta_yaw=secure_random.uniform(-0.02, 0.02),
    )

    return {
        "image": image,
        "instruction": secure_random.choice(TAKEOFF_INSTRUCTIONS),
        "action": action,
    }


def demonstrations_to_tensors(
    demonstrations: list[dict],
    device: torch.device,
) -> tuple[list[Image.Image], list[str], Tensor]:
    images = [d["image"] for d in demonstrations]
    instructions = [d["instruction"] for d in demonstrations]

    actions = []
    for d in demonstrations:
        action = d["action"]
        actions.append(
            [
                action.delta_x,
                action.delta_y,
                action.delta_z,
                action.delta_roll,
                action.delta_pitch,
                action.delta_yaw,
            ],
        )

    action_tensor = torch.tensor(actions, dtype=torch.float32, device=device)

    return images, instructions, action_tensor
