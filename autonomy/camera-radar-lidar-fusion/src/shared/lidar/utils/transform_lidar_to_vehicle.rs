use anyhow::Result;
use nalgebra::{Matrix4, Matrix4xX, Vector4};

pub fn transform_lidar_to_vehicle(
    lidar_points: &Matrix4xX<f32>,
    lidar_to_vehicle_transform: &Matrix4<f64>,
) -> Result<Matrix4xX<f32>> {
    let transform_f32: Matrix4<f32> = lidar_to_vehicle_transform.map(|value| value as f32);
    let point_count = lidar_points.ncols();
    let mut transformed_columns = Vec::with_capacity(point_count);

    for column_index in 0..point_count {
        let column = lidar_points.column(column_index);

        // Convert to homogeneous coordinates (x, y, z, 1)
        let point_lidar = Vector4::new(column[0], column[1], column[2], 1.0);

        // Transform to vehicle frame
        let point_vehicle = transform_f32 * point_lidar;

        // Keep intensity from original data
        let intensity = column[3];

        // Store transformed point with intensity
        transformed_columns.push(Vector4::new(
            point_vehicle[0],
            point_vehicle[1],
            point_vehicle[2],
            intensity,
        ));
    }

    Ok(Matrix4xX::from_columns(&transformed_columns))
}
