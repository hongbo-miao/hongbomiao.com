import logging

from isaacsim import SimulationApp

logger = logging.getLogger(__name__)

SIMULATION_CONFIG = {
    "headless": False,
    "width": 1280,
    "height": 720,
    "anti_aliasing": 1,
}


def create_quadrotor_environment(
    config: dict | None = None,
    headless: bool = False,
) -> SimulationApp:
    """
    Create Isaac Sim environment for quadrotor simulation.

    Args:
        config: Optional simulation configuration overrides
        headless: Run without GUI window (default: False to enable visualization)

    Returns:
        SimulationApp instance

    """
    simulation_config = SIMULATION_CONFIG.copy()
    simulation_config["headless"] = headless
    if config:
        simulation_config.update(config)

    logger.info("Initializing Isaac Sim environment")
    simulation_app = SimulationApp(simulation_config)

    from omni.isaac.core import World  # noqa: PLC0415
    from omni.isaac.core.utils.nucleus import get_assets_root_path  # noqa: PLC0415
    from omni.isaac.core.utils.stage import add_reference_to_stage  # noqa: PLC0415

    world = World(stage_units_in_meters=1.0)

    assets_root_path = get_assets_root_path()
    logger.info(f"Assets root path: {assets_root_path}")

    add_reference_to_stage(
        usd_path=f"{assets_root_path}/Isaac/Environments/Simple_Warehouse/warehouse.usd",
        prim_path="/World/Environment",
    )

    world.scene.add_default_ground_plane()

    logger.info("Isaac Sim environment created successfully")

    return simulation_app
