use crate::shared::map::types::ego_pose::EgoPose;
use crate::shared::map::utils::derive_latitude_longitude::derive_latitude_longitude;
use anyhow::Result;
use rerun as rr;

pub fn log_ego_position_to_rerun(
    recording: &rr::RecordingStream,
    ego_pose: &EgoPose,
    location: &str,
    entity_path: &str,
) -> Result<()> {
    let x = ego_pose.translation[0];
    let y = ego_pose.translation[1];

    if let Some((latitude, longitude)) = derive_latitude_longitude(location, x, y) {
        recording.log(
            entity_path,
            &rr::GeoPoints::from_lat_lon([[latitude, longitude]])
                .with_radii([rr::Radius::new_ui_points(3.0)])
                .with_colors([rr::components::Color::from_rgb(255, 0, 0)]),
        )?;
    }

    Ok(())
}
