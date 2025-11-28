use anyhow::Result;
use nalgebra::Matrix5xX;
use rerun as rr;

pub fn log_radar_to_rerun(
    recording: &rr::RecordingStream,
    radar_points: &Matrix5xX<f32>,
    entity_path: &str,
) -> Result<()> {
    let point_count = radar_points.ncols();

    if point_count == 0 {
        return Ok(());
    }

    // Extract x, y, z coordinates
    let mut positions = Vec::with_capacity(point_count);
    let mut colors = Vec::with_capacity(point_count);

    for column in radar_points.column_iter() {
        let x = column[0];
        let y = column[1];
        let z = column[2];
        let velocity = column[3];
        let _radar_cross_section = column[4];

        positions.push([x, y, z]);

        // Color based on velocity (red for approaching, blue for receding)
        let normalized_velocity = (velocity / 20.0).clamp(-1.0, 1.0);

        if normalized_velocity > 0.0 {
            // Approaching (positive velocity) - red gradient
            let intensity = (normalized_velocity * 255.0) as u8;
            colors.push([255, 255 - intensity, 255 - intensity, 255]);
        } else {
            // Receding (negative velocity) - blue gradient
            let intensity = (normalized_velocity.abs() * 255.0) as u8;
            colors.push([255 - intensity, 255 - intensity, 255, 255]);
        }
    }

    recording.log(
        entity_path,
        &rr::Points3D::new(positions)
            .with_colors(colors)
            .with_radii([0.1]), // 10cm radius for radar points (larger than lidar)
    )?;

    Ok(())
}
