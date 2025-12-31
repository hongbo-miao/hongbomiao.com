import logging
from pathlib import Path

import torch
from shared.simulation.utils.apply_action_to_quadrotor import apply_action_to_quadrotor
from shared.simulation.utils.create_camera import create_camera, update_camera_position
from shared.simulation.utils.create_quadrotor import create_quadrotor
from shared.simulation.utils.create_quadrotor_environment import (
    create_quadrotor_environment,
)
from shared.simulation.utils.get_camera_observation import get_camera_observation
from shared.simulation.utils.get_vehicle_state import get_vehicle_state
from shared.visualization.utils.plot_trajectory import plot_trajectory
from shared.visualization.utils.save_camera_observation import save_camera_observation

logger = logging.getLogger(__name__)

DINOV3_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
QWEN3_MODEL_ID = "Qwen/Qwen3-0.6B"
CHECKPOINT_DIRECTORY = Path("../train/output/checkpoints")
FLOW_MATCHING_POLICY_CHECKPOINT_PATH = CHECKPOINT_DIRECTORY / "flow_matching_policy.pt"
VISION_PROJECTION_CHECKPOINT_PATH = CHECKPOINT_DIRECTORY / "vision_projection.pt"

INSTRUCTION = "To the moon!"
MAX_EPISODE_STEPS = 200
OUTPUT_DIRECTORY = Path("output")


def run_vla_episode(
    instruction: str = INSTRUCTION,
    max_steps: int = MAX_EPISODE_STEPS,
    headless: bool = False,
    save_images: bool = True,
    save_every_n_steps: int = 5,
) -> None:
    logger.info(f"Starting VLA episode with instruction: {instruction}")
    logger.info(f"Visualization: headless={headless}, save_images={save_images}")

    logger.info("Creating Isaac Sim environment")
    simulation_app = create_quadrotor_environment(headless=headless)

    from omni.isaac.core import World  # noqa: PLC0415

    world = World.instance()
    world.reset()

    logger.info("Creating quadrotor")
    create_quadrotor(
        prim_path="/World/Quadrotor",
        position=(0.0, 0.0, 2.0),
    )

    logger.info("Creating camera")
    camera = create_camera(
        camera_prim_path="/World/Camera",
        resolution=(640, 480),
        position_offset=(0.0, -8.0, 5.0),
    )

    world.reset()

    camera.initialize()

    for _ in range(20):
        world.step(render=True)

    logger.info("Initializing VLA agent (loading models...)")
    from shared.agent.utils.create_vla_agent import create_vla_agent  # noqa: PLC0415

    # Force CPU because Isaac Sim relies on an old PyTorch version
    # that does not support RTX 5090 (sm_120 Blackwell)
    device = torch.device("cpu")

    agent = create_vla_agent(
        vision_model_id=DINOV3_MODEL_ID,
        language_model_id=QWEN3_MODEL_ID,
        flow_matching_policy_checkpoint_path=FLOW_MATCHING_POLICY_CHECKPOINT_PATH,
        vision_projection_checkpoint_path=VISION_PROJECTION_CHECKPOINT_PATH,
        flow_matching_policy_action_dimension=6,
        flow_matching_policy_hidden_dimension=2048,
        flow_matching_policy_layer_count=8,
        device=device,
    )

    current_state = get_vehicle_state()

    trajectory_positions: list[tuple[float, float, float]] = []
    trajectory_positions.append(
        (
            current_state.position_x,
            current_state.position_y,
            current_state.position_z,
        ),
    )

    image_output_directory = OUTPUT_DIRECTORY / "frames"

    logger.info("Starting control loop")
    for step in range(max_steps):
        image = get_camera_observation(camera=camera)

        if save_images and step % save_every_n_steps == 0:
            save_camera_observation(
                image=image,
                output_directory=image_output_directory,
                step=step,
            )

        action = agent.predict_action(
            image=image,
            instruction=instruction,
        )

        current_state = apply_action_to_quadrotor(
            action=action,
            current_state=current_state,
        )

        world.step(render=True)

        update_camera_position(
            camera=camera,
            target_position=(
                current_state.position_x,
                current_state.position_y,
                current_state.position_z,
            ),
            offset=(-2.0, -2.0, 1.0),
        )

        trajectory_positions.append(
            (
                current_state.position_x,
                current_state.position_y,
                current_state.position_z,
            ),
        )

        if step % 10 == 0:
            logger.info(
                f"Step {step}: position=({current_state.position_x:.2f}, "
                f"{current_state.position_y:.2f}, {current_state.position_z:.2f})",
            )

        if simulation_app.is_exiting():
            logger.info("Simulation app is exiting")
            break

    logger.info("Episode completed")

    logger.info("Generating trajectory plot...")
    trajectory_filename = f"trajectory_{instruction.replace(' ', '_')}.png"
    plot_trajectory(
        positions=trajectory_positions,
        output_path=OUTPUT_DIRECTORY / trajectory_filename,
        title=f"VLA Episode: {instruction}",
    )

    if save_images:
        logger.info(f"Camera frames saved to: {image_output_directory}")

    simulation_app.close()


def main() -> None:
    run_vla_episode(
        instruction=INSTRUCTION,
        max_steps=MAX_EPISODE_STEPS,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
