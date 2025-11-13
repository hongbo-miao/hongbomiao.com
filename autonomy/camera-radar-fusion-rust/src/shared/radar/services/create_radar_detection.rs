use crate::shared::radar::types::radar_detection::RadarDetection;
use nalgebra::Vector3;

pub fn create_radar_detection(
    position_3d: Vector3<f64>,
    velocity: f64,
    radar_cross_section: f64,
    image_coordinate_x: f64,
    image_coordinate_y: f64,
) -> RadarDetection {
    RadarDetection::new(
        position_3d,
        velocity,
        radar_cross_section,
        image_coordinate_x,
        image_coordinate_y,
    )
}
