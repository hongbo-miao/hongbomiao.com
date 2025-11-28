use crate::shared::occupancy::types::occupancy_grid::OccupancyGrid;
use nalgebra::Vector3;

pub fn clear_distant_voxels(occupancy_grid: &mut OccupancyGrid, max_distance: f32) {
    let origin = Vector3::new(0.0, 0.0, 0.0);
    let max_distance_squared = max_distance * max_distance;

    occupancy_grid.voxels.retain(|_, voxel| {
        let distance_squared = (voxel.center - origin).norm_squared();
        distance_squared <= max_distance_squared
    });
}
