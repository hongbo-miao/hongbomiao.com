use nalgebra::{Matrix3, Matrix4, Vector2, Vector3};

pub fn project_lidar_to_camera(
    lidar_points: &[Vector3<f64>],
    camera_intrinsic: &Matrix3<f64>,
    lidar_to_camera_transform: &Matrix4<f64>,
) -> Vec<Option<Vector2<f64>>> {
    if lidar_points.is_empty() {
        return Vec::new();
    }

    let mut image_points = Vec::with_capacity(lidar_points.len());

    for lidar_point in lidar_points {
        let homogeneous = lidar_point.push(1.0);
        let camera_point = lidar_to_camera_transform * homogeneous;

        if camera_point.z > 0.0 {
            let point_3d = camera_point.xyz();
            let projected = camera_intrinsic * point_3d;

            let u = projected.x / projected.z;
            let v = projected.y / projected.z;

            image_points.push(Some(Vector2::new(u, v)));
        } else {
            image_points.push(None);
        }
    }

    image_points
}
