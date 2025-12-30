import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def plot_trajectory(
    positions: list[tuple[float, float, float]],
    output_path: Path | None = None,
    title: str = "Quadrotor Trajectory",
) -> None:
    """
    Plot 3D trajectory of the quadrotor.

    Args:
        positions: List of (x, y, z) positions over time
        output_path: Path to save plot (displays interactively if None)
        title: Plot title

    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if len(positions) < 2:
        logger.warning("Not enough positions to plot trajectory")
        return

    x_values = [p[0] for p in positions]
    y_values = [p[1] for p in positions]
    z_values = [p[2] for p in positions]

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot(x_values, y_values, z_values, "b-", linewidth=2, label="Trajectory")
    ax1.scatter(
        x_values[0],
        y_values[0],
        z_values[0],
        c="green",
        s=100,
        label="Start",
        marker="o",
    )
    ax1.scatter(
        x_values[-1],
        y_values[-1],
        z_values[-1],
        c="red",
        s=100,
        label="End",
        marker="x",
    )
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title(f"{title} - 3D View")
    ax1.legend()

    ax2 = fig.add_subplot(122)
    step_count = len(positions)
    steps = list(range(step_count))
    ax2.plot(steps, x_values, "r-", label="X")
    ax2.plot(steps, y_values, "g-", label="Y")
    ax2.plot(steps, z_values, "b-", label="Z")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Position (m)")
    ax2.set_title(f"{title} - Position over Time")
    ax2.legend()
    ax2.grid(visible=True)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        logger.info(f"Saved trajectory plot: {output_path}")
    else:
        plt.show()

    plt.close()
