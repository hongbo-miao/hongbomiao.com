import logging
from pathlib import Path

from nuscenes.nuscenes import NuScenes
from shared.fusion.services.visualize_camera_radar_fusion import (
    visualize_camera_radar_fusion,
)

logger = logging.getLogger(__name__)


def main() -> None:
    dataset_path = Path("data/v1.0-mini")

    if not dataset_path.exists():
        logger.error(f"nuScenes dataset not found at {dataset_path}")
        return

    # Load dataset
    logger.info(f"Loading nuScenes dataset from {dataset_path}")
    nuscenes_instance = NuScenes(
        version="v1.0-mini",
        dataroot=str(dataset_path),
        verbose=True,
    )

    logger.info(f"Dataset loaded: {len(nuscenes_instance.scene)} scenes available")
    logger.info("Controls: 'q' = quit, 'space' = pause")

    # Visualize first scene
    visualize_camera_radar_fusion(nuscenes_instance, scene_index=0, max_frames=100)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
