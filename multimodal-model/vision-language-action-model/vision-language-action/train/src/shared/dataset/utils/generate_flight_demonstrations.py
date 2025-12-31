import logging
from random import SystemRandom

import torch
from PIL import Image
from shared.image.utils.create_altitude_image import create_altitude_image
from torch import Tensor
from vision_language_action_lib.types.action_output import ActionOutput

random_number_generator = SystemRandom()

logger = logging.getLogger(__name__)


LANDING_INSTRUCTIONS = [
    # Short commands
    "landing",
    "land",
    "descend",
    "go down",
    "land on the ground",
    "descend to ground",
    "lower altitude",
    # Sentence-based instructions
    "Please land the quadrotor safely.",
    "Descend and land on the ground below.",
    "Bring the aircraft down for landing.",
    "Start the landing procedure now.",
    "Lower the altitude and prepare to land.",
    "Reduce altitude and touch down gently.",
    "Begin descent to the landing zone.",
    "I need you to land the drone.",
    "Can you land the quadrotor here?",
    "Execute a controlled descent to the surface.",
]

TAKEOFF_INSTRUCTIONS = [
    # Short commands
    "take off",
    "takeoff",
    "go up",
    "ascend",
    "rise",
    "climb",
    "increase altitude",
    "fly up",
    # Sentence-based instructions
    "Please take off from the current position.",
    "Ascend to a higher altitude.",
    "Lift off and gain some height.",
    "Start the takeoff sequence now.",
    "Increase your altitude immediately.",
    "Rise up from the ground.",
    "Begin the ascent procedure.",
    "I need you to fly upward.",
    "Can you take the quadrotor higher?",
    "Execute a vertical climb from here.",
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

    random_number_generator.shuffle(demonstrations)

    logger.info(
        f"Generated {sample_count} flight demonstrations ({landing_count} landing, {takeoff_count} takeoff)",
    )
    return demonstrations


def generate_landing_demonstration(
    image_size: tuple[int, int] = (256, 256),
) -> dict[str, Image.Image | str | ActionOutput]:
    simulated_altitude_m = random_number_generator.uniform(450.0, 1200.0)

    # quadrotor descent rates: ~4-5 m/s at cruise altitude, ~1-2 m/s on final approach
    if simulated_altitude_m > 900.0:
        descent_rate_mps = random_number_generator.uniform(-5.0, -4.0)
    elif simulated_altitude_m > 600.0:
        descent_rate_mps = random_number_generator.uniform(-3.5, -2.5)
    else:
        descent_rate_mps = random_number_generator.uniform(-2.0, -1.0)

    image = create_altitude_image(
        simulated_altitude_m=simulated_altitude_m,
        image_size=image_size,
    )

    action = ActionOutput(
        delta_x_mps=random_number_generator.uniform(-0.5, 0.5),
        delta_y_mps=random_number_generator.uniform(-0.5, 0.5),
        delta_z_mps=descent_rate_mps,
        delta_roll_radps=random_number_generator.uniform(-0.05, 0.05),
        delta_pitch_radps=random_number_generator.uniform(-0.05, 0.05),
        delta_yaw_radps=random_number_generator.uniform(-0.05, 0.05),
    )

    return {
        "image": image,
        "instruction": random_number_generator.choice(LANDING_INSTRUCTIONS),
        "action": action,
    }


def generate_takeoff_demonstration(
    image_size: tuple[int, int] = (256, 256),
) -> dict[str, Image.Image | str | ActionOutput]:
    simulated_altitude_m = random_number_generator.uniform(0.0, 600.0)

    # quadrotor climb rates: ~6-8 m/s initial climb, ~3-4 m/s toward cruise altitude
    if simulated_altitude_m < 150.0:
        ascent_rate_mps = random_number_generator.uniform(6.0, 8.0)
    elif simulated_altitude_m < 450.0:
        ascent_rate_mps = random_number_generator.uniform(4.0, 6.0)
    else:
        ascent_rate_mps = random_number_generator.uniform(2.0, 4.0)

    image = create_altitude_image(
        simulated_altitude_m=simulated_altitude_m,
        image_size=image_size,
    )

    action = ActionOutput(
        delta_x_mps=random_number_generator.uniform(-0.5, 0.5),
        delta_y_mps=random_number_generator.uniform(-0.5, 0.5),
        delta_z_mps=ascent_rate_mps,
        delta_roll_radps=random_number_generator.uniform(-0.05, 0.05),
        delta_pitch_radps=random_number_generator.uniform(-0.05, 0.05),
        delta_yaw_radps=random_number_generator.uniform(-0.05, 0.05),
    )

    return {
        "image": image,
        "instruction": random_number_generator.choice(TAKEOFF_INSTRUCTIONS),
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
                action.delta_x_mps,
                action.delta_y_mps,
                action.delta_z_mps,
                action.delta_roll_radps,
                action.delta_pitch_radps,
                action.delta_yaw_radps,
            ],
        )

    action_tensor = torch.tensor(actions, dtype=torch.float32, device=device)

    return images, instructions, action_tensor
