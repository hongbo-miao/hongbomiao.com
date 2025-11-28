use anyhow::Result;
use nalgebra::Matrix4xX;
use rerun as rr;

pub fn log_lidar_to_rerun(
    recording: &rr::RecordingStream,
    lidar_points: &Matrix4xX<f32>,
    entity_path: &str,
) -> Result<()> {
    let point_count = lidar_points.ncols();

    if point_count == 0 {
        return Ok(());
    }

    // Extract x, y, z coordinates
    let mut positions = Vec::with_capacity(point_count);
    let mut colors = Vec::with_capacity(point_count);

    for column in lidar_points.column_iter() {
        let x = column[0];
        let y = column[1];
        let z = column[2];
        let intensity = column[3];

        positions.push([x, y, z]);

        // Color based on intensity (0-255 range typically)
        let normalized_intensity = (intensity / 255.0).clamp(0.0, 1.0);
        let color_value = (normalized_intensity * 255.0) as u8;
        colors.push([color_value, color_value, color_value, 255]);
    }

    recording.log(
        entity_path,
        &rr::Points3D::new(positions)
            .with_colors(colors)
            .with_radii([0.05]), // 5cm radius for each point
    )?;

    Ok(())
}
