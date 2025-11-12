import logging
from pathlib import Path

from config import config
from nuscenes.nuscenes import NuScenes
from shared.fusion.services.visualize_camera_radar_fusion import (
    visualize_camera_radar_fusion,
)

logger = logging.getLogger(__name__)


def main() -> None:
    dataset_path = Path(config.NUSCENES_DATASET_PATH)

    if not dataset_path.exists():
        logger.error(f"nuScenes dataset not found at {dataset_path}")
        return

    # Load dataset
    logger.info(f"Loading nuScenes dataset from {dataset_path}")
    nuscenes_instance = NuScenes(
        version=config.NUSCENES_VERSION,
        dataroot=str(dataset_path),
        verbose=True,
    )

    logger.info(f"Dataset loaded: {len(nuscenes_instance.scene)} scenes available")
    logger.info("Controls: 'q' = quit, 'space' = pause")

    # Visualize scene
    visualize_camera_radar_fusion(
        nuscenes_instance,
        nuscenes_scene_index=config.NUSCENES_SCENE_INDEX,
        visualization_frame_count=config.VISUALIZATION_FRAME_COUNT,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
