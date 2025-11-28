use crate::shared::occupancy::types::occupancy_grid::{OccupancyGrid, VoxelState};

pub fn decay_occupancy_probabilities(occupancy_grid: &mut OccupancyGrid, decay_rate: f32) {
    for voxel in occupancy_grid.voxels.values_mut() {
        // Decay towards 0.5 (unknown state)
        if voxel.occupancy_probability > 0.5 {
            voxel.occupancy_probability = (voxel.occupancy_probability - decay_rate).max(0.5);
        } else if voxel.occupancy_probability < 0.5 {
            voxel.occupancy_probability = (voxel.occupancy_probability + decay_rate).min(0.5);
        }

        // Update state based on new probability
        if voxel.occupancy_probability >= occupancy_grid.config.occupied_threshold {
            voxel.state = VoxelState::Occupied;
        } else if voxel.occupancy_probability <= occupancy_grid.config.free_threshold {
            voxel.state = VoxelState::Free;
        } else {
            voxel.state = VoxelState::Unknown;
        }
    }
}
