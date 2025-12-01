use crate::shared::occupancy::types::occupancy_grid::OccupancyGrid;
use anyhow::Result;
use nalgebra::{Matrix4xX, Vector3};

pub fn build_occupancy_grid_from_lidar(
    occupancy_grid: &mut OccupancyGrid,
    lidar_points: &Matrix4xX<f32>,
    sensor_origin: &Vector3<f32>,
) -> Result<()> {
    let point_count = lidar_points.ncols();

    for column_index in 0..point_count {
        let column = lidar_points.column(column_index);
        let point = Vector3::new(column[0], column[1], column[2]);

        if !occupancy_grid.is_within_bounds(&point) {
            continue;
        }

        // Cache config values to avoid overlapping borrows
        let occupied_threshold = occupancy_grid.config.occupied_threshold;
        let free_threshold = occupancy_grid.config.free_threshold;
        let occupied_probability_given_occupied_evidence = occupancy_grid
            .config
            .occupied_probability_given_occupied_evidence;
        let occupied_probability_given_free_evidence = occupancy_grid
            .config
            .occupied_probability_given_free_evidence;

        // Mark the voxel containing the point as occupied
        let voxel_index = occupancy_grid.world_to_voxel_index(&point);
        let voxel = occupancy_grid.get_or_create_voxel(voxel_index);
        voxel.update_occupancy(
            true,
            occupied_threshold,
            free_threshold,
            occupied_probability_given_occupied_evidence,
            occupied_probability_given_free_evidence,
        );

        // Ray casting: mark free space from sensor origin to the point
        mark_free_space_along_ray(occupancy_grid, sensor_origin, &point)?;
    }

    Ok(())
}

fn mark_free_space_along_ray(
    occupancy_grid: &mut OccupancyGrid,
    origin: &Vector3<f32>,
    end_point: &Vector3<f32>,
) -> Result<()> {
    let ray_direction = end_point - origin;
    let ray_length = ray_direction.norm();
    let ray_direction_normalized = ray_direction.normalize();

    let step_size = occupancy_grid.config.voxel_size * 0.5;
    let step_count = (ray_length / step_size).floor() as usize;

    for step_index in 0..step_count {
        let current_position = origin + ray_direction_normalized * (step_index as f32 * step_size);

        if !occupancy_grid.is_within_bounds(&current_position) {
            continue;
        }

        let voxel_index = occupancy_grid.world_to_voxel_index(&current_position);

        // Cache config values to avoid overlapping borrows
        let occupied_threshold = occupancy_grid.config.occupied_threshold;
        let free_threshold = occupancy_grid.config.free_threshold;
        let occupied_probability_given_occupied_evidence = occupancy_grid
            .config
            .occupied_probability_given_occupied_evidence;
        let occupied_probability_given_free_evidence = occupancy_grid
            .config
            .occupied_probability_given_free_evidence;

        // Don't overwrite occupied voxels
        if let Some(existing_voxel) = occupancy_grid.get_voxel(&voxel_index)
            && existing_voxel.occupancy_probability > occupied_threshold
        {
            continue;
        }

        let voxel = occupancy_grid.get_or_create_voxel(voxel_index);
        voxel.update_occupancy(
            false,
            occupied_threshold,
            free_threshold,
            occupied_probability_given_occupied_evidence,
            occupied_probability_given_free_evidence,
        );
    }

    Ok(())
}
