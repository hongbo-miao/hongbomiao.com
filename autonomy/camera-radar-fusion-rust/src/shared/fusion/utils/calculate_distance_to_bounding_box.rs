use crate::shared::camera::types::camera_detection::CameraDetection;
use crate::shared::fusion::utils::is_radar_inside_bounding_box::is_radar_inside_bounding_box;
use crate::shared::radar::types::radar_detection::RadarDetection;

pub fn calculate_distance_to_bounding_box(
    radar_detection: &RadarDetection,
    camera_detection: &CameraDetection,
) -> f64 {
    if is_radar_inside_bounding_box(radar_detection, camera_detection) {
        return 0.0;
    }

    let bounding_box = &camera_detection.bounding_box;
    let x1 = bounding_box[0];
    let y1 = bounding_box[1];
    let x2 = bounding_box[2];
    let y2 = bounding_box[3];

    let radar_x = radar_detection.image_coordinate_x;
    let radar_y = radar_detection.image_coordinate_y;

    let closest_x = radar_x.clamp(x1, x2);
    let closest_y = radar_y.clamp(y1, y2);

    ((radar_x - closest_x).powi(2) + (radar_y - closest_y).powi(2)).sqrt()
}
