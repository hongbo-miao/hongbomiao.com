use crate::shared::lidar::types::lidar_detection::LidarDetection;
use nalgebra::Vector3;

pub fn create_lidar_detection(
    position_3d: Vector3<f64>,
    intensity: f64,
    image_coordinate_x: f64,
    image_coordinate_y: f64,
) -> LidarDetection {
    LidarDetection::new(
        position_3d,
        intensity,
        image_coordinate_x,
        image_coordinate_y,
    )
}
