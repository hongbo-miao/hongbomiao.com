import logging

import genesis as gs
import numpy as np

logger = logging.getLogger(__name__)


def main() -> None:
    # Initialize Genesis with GPU backend
    gs.init(backend=gs.gpu)

    # Create scene with camera and simulation settings
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -3.5, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        show_viewer=True,
    )

    # Add plane and Franka robot arm to scene
    scene.add_entity(
        gs.morphs.Plane(),
    )
    franka_robot_arm = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )

    # Build scene
    scene.build()

    jnt_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "finger_joint1",
        "finger_joint2",
    ]
    dofs_idx = [franka_robot_arm.get_joint(name).dof_idx_local for name in jnt_names]

    # Configure robot control parameters
    # Set positional gains
    franka_robot_arm.set_dofs_kp(
        kp=np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        dofs_idx_local=dofs_idx,
    )
    # Set velocity gains
    franka_robot_arm.set_dofs_kv(
        kv=np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        dofs_idx_local=dofs_idx,
    )
    # Set force range for safety
    franka_robot_arm.set_dofs_force_range(
        lower=np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        upper=np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        dofs_idx_local=dofs_idx,
    )

    # Hard reset sequence
    for i in range(150):
        if i < 50:
            franka_robot_arm.set_dofs_position(
                np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]),
                dofs_idx,
            )
        elif i < 100:
            franka_robot_arm.set_dofs_position(
                np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]),
                dofs_idx,
            )
        else:
            franka_robot_arm.set_dofs_position(
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                dofs_idx,
            )

        scene.step()

    # PD control sequence
    for i in range(1250):
        if i == 0:
            franka_robot_arm.control_dofs_position(
                np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]),
                dofs_idx,
            )
        elif i == 250:
            franka_robot_arm.control_dofs_position(
                np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]),
                dofs_idx,
            )
        elif i == 500:
            franka_robot_arm.control_dofs_position(
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                dofs_idx,
            )
        elif i == 750:
            # control first dof with velocity, and the rest with position
            franka_robot_arm.control_dofs_position(
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])[1:],
                dofs_idx[1:],
            )
            franka_robot_arm.control_dofs_velocity(
                np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0])[:1],
                dofs_idx[:1],
            )
        elif i == 1000:
            franka_robot_arm.control_dofs_force(
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                dofs_idx,
            )
        # This is the control force computed based on the given control command
        # If using force control, it's the same as the given control command
        logger.info(
            f"control force: {franka_robot_arm.get_dofs_control_force(dofs_idx)}",
        )

        # This is the actual force experienced by the dof
        logger.info(f"internal force: {franka_robot_arm.get_dofs_force(dofs_idx)}")

        scene.step()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
