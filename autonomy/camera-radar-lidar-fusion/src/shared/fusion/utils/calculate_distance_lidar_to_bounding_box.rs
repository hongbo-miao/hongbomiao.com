use crate::shared::camera::types::camera_detection::CameraDetection;
use crate::shared::fusion::utils::is_lidar_inside_bounding_box::is_lidar_inside_bounding_box;
use crate::shared::lidar::types::lidar_detection::LidarDetection;

pub fn calculate_distance_lidar_to_bounding_box(
    lidar_detection: &LidarDetection,
    camera_detection: &CameraDetection,
) -> f64 {
    if is_lidar_inside_bounding_box(lidar_detection, camera_detection) {
        return 0.0;
    }

    let bounding_box = &camera_detection.bounding_box;
    let x1 = bounding_box[0];
    let y1 = bounding_box[1];
    let x2 = bounding_box[2];
    let y2 = bounding_box[3];

    let lidar_x = lidar_detection.image_coordinate_x;
    let lidar_y = lidar_detection.image_coordinate_y;

    let closest_x = lidar_x.clamp(x1, x2);
    let closest_y = lidar_y.clamp(y1, y2);

    ((lidar_x - closest_x).powi(2) + (lidar_y - closest_y).powi(2)).sqrt()
}
