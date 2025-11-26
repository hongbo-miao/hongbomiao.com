use crate::shared::camera::types::camera_detection::CameraDetection;
use crate::shared::radar::types::radar_detection::RadarDetection;

pub fn is_radar_inside_bounding_box(
    radar_detection: &RadarDetection,
    camera_detection: &CameraDetection,
) -> bool {
    let bounding_box = &camera_detection.bounding_box;
    let x1 = bounding_box[0];
    let y1 = bounding_box[1];
    let x2 = bounding_box[2];
    let y2 = bounding_box[3];

    x1 <= radar_detection.image_coordinate_x
        && radar_detection.image_coordinate_x <= x2
        && y1 <= radar_detection.image_coordinate_y
        && radar_detection.image_coordinate_y <= y2
}
