use crate::shared::occupancy::types::occupancy_grid::{OccupancyGrid, VoxelState};
use anyhow::Result;
use rerun as rr;

pub fn log_occupancy_grid_to_rerun(
    recording: &rr::RecordingStream,
    occupancy_grid: &OccupancyGrid,
    entity_path: &str,
) -> Result<()> {
    let mut occupied_centers = Vec::new();
    let mut occupied_colors = Vec::new();
    let mut free_centers = Vec::new();
    let mut unknown_centers = Vec::new();

    let half_size = occupancy_grid.config.voxel_size / 2.0;

    for voxel in occupancy_grid.voxels.values() {
        let center = [voxel.center.x, voxel.center.y, voxel.center.z];

        match voxel.state {
            VoxelState::Occupied => {
                occupied_centers.push(center);

                // Color by occupancy probability: darker = more certain
                let intensity = (voxel.occupancy_probability * 255.0) as u8;
                occupied_colors.push([intensity, 0, 0, 200]); // Red for occupied
            }
            VoxelState::Free => {
                free_centers.push(center);
            }
            VoxelState::Unknown => {
                unknown_centers.push(center);
            }
        }
    }

    // Log occupied voxels (red)
    if !occupied_centers.is_empty() {
        recording.log(
            format!("{}/occupied", entity_path),
            &rr::Boxes3D::from_half_sizes(vec![
                [half_size, half_size, half_size];
                occupied_centers.len()
            ])
            .with_centers(occupied_centers)
            .with_colors(occupied_colors),
        )?;
    }

    // Log free voxels (green, semi-transparent)
    if !free_centers.is_empty() {
        let free_voxel_count = free_centers.len();
        recording.log(
            format!("{}/free", entity_path),
            &rr::Boxes3D::from_half_sizes(vec![
                [half_size, half_size, half_size];
                free_voxel_count
            ])
            .with_centers(free_centers)
            .with_colors(vec![[0, 255, 0, 50]; free_voxel_count]), // Green, very transparent
        )?;
    }

    // Log unknown voxels (yellow, semi-transparent)
    if !unknown_centers.is_empty() {
        let unknown_voxel_count = unknown_centers.len();
        recording.log(
            format!("{}/unknown", entity_path),
            &rr::Boxes3D::from_half_sizes(vec![
                [half_size, half_size, half_size];
                unknown_voxel_count
            ])
            .with_centers(unknown_centers)
            .with_colors(vec![[255, 255, 0, 100]; unknown_voxel_count]), // Yellow
        )?;
    }

    Ok(())
}
