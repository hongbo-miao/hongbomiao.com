use crate::shared::map::types::ego_pose::EgoPose;
use crate::shared::map::utils::derive_latitude_longitude::derive_latitude_longitude;
use anyhow::Result;
use rerun as rr;

pub fn log_ego_trajectory_to_rerun(
    recording: &rr::RecordingStream,
    ego_poses: &[&EgoPose],
    location: &str,
    entity_path: &str,
) -> Result<()> {
    if ego_poses.is_empty() {
        return Ok(());
    }

    let mut latitude_longitude_points = Vec::new();

    for ego_pose in ego_poses {
        let x = ego_pose.translation[0];
        let y = ego_pose.translation[1];

        if let Some((latitude, longitude)) = derive_latitude_longitude(location, x, y) {
            latitude_longitude_points.push([latitude, longitude]);
        }
    }

    if latitude_longitude_points.is_empty() {
        anyhow::bail!("No valid GPS coordinates could be derived from ego poses");
    }

    recording.log(
        entity_path,
        &rr::GeoLineStrings::from_lat_lon([latitude_longitude_points])
            .with_radii([rr::Radius::new_ui_points(2.0)])
            .with_colors([rr::components::Color::from_rgb(255, 0, 0)]),
    )?;

    Ok(())
}
