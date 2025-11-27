use crate::shared::camera::types::camera_detection::CameraDetection;
use crate::shared::lidar::types::lidar_detection::LidarDetection;

pub fn is_lidar_inside_bounding_box(
    lidar_detection: &LidarDetection,
    camera_detection: &CameraDetection,
) -> bool {
    let bounding_box = &camera_detection.bounding_box;
    let x1 = bounding_box[0];
    let y1 = bounding_box[1];
    let x2 = bounding_box[2];
    let y2 = bounding_box[3];

    x1 <= lidar_detection.image_coordinate_x
        && lidar_detection.image_coordinate_x <= x2
        && y1 <= lidar_detection.image_coordinate_y
        && lidar_detection.image_coordinate_y <= y2
}
