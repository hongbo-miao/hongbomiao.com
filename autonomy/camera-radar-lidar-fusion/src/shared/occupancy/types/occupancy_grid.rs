use nalgebra::Vector3;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VoxelState {
    Free,
    Occupied,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct Voxel {
    pub state: VoxelState,
    pub occupancy_probability: f32,
    pub center: Vector3<f32>,
}

impl Voxel {
    pub fn new(center: Vector3<f32>) -> Self {
        Self {
            state: VoxelState::Unknown,
            occupancy_probability: 0.5,
            center,
        }
    }

    pub fn update_occupancy(
        &mut self,
        is_occupied: bool,
        occupied_threshold: f32,
        free_threshold: f32,
        occupied_probability_increment: f32,
        free_probability_decrement: f32,
    ) {
        if is_occupied {
            self.occupancy_probability =
                (self.occupancy_probability + occupied_probability_increment).min(1.0);
        } else {
            self.occupancy_probability =
                (self.occupancy_probability - free_probability_decrement).max(0.0);
        }

        if self.occupancy_probability >= occupied_threshold {
            self.state = VoxelState::Occupied;
        } else if self.occupancy_probability <= free_threshold {
            self.state = VoxelState::Free;
        } else {
            self.state = VoxelState::Unknown;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoxelIndex {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl VoxelIndex {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

#[derive(Debug, Clone)]
pub struct OccupancyGridConfig {
    pub voxel_size: f32,
    pub min_bound: Vector3<f32>,
    pub max_bound: Vector3<f32>,
    pub occupied_threshold: f32,
    pub free_threshold: f32,
    pub occupied_probability_increment: f32,
    pub free_probability_decrement: f32,
}

impl OccupancyGridConfig {
    pub fn new(
        voxel_size: f32,
        min_bound: Vector3<f32>,
        max_bound: Vector3<f32>,
        occupied_threshold: f32,
        free_threshold: f32,
        occupied_probability_increment: f32,
        free_probability_decrement: f32,
    ) -> Self {
        Self {
            voxel_size,
            min_bound,
            max_bound,
            occupied_threshold,
            free_threshold,
            occupied_probability_increment,
            free_probability_decrement,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OccupancyGrid {
    pub config: OccupancyGridConfig,
    pub voxels: HashMap<VoxelIndex, Voxel>,
}

impl OccupancyGrid {
    pub fn new(config: OccupancyGridConfig) -> Self {
        Self {
            config,
            voxels: HashMap::new(),
        }
    }

    pub fn world_to_voxel_index(&self, position: &Vector3<f32>) -> VoxelIndex {
        let x = ((position.x - self.config.min_bound.x) / self.config.voxel_size).floor() as i32;
        let y = ((position.y - self.config.min_bound.y) / self.config.voxel_size).floor() as i32;
        let z = ((position.z - self.config.min_bound.z) / self.config.voxel_size).floor() as i32;
        VoxelIndex::new(x, y, z)
    }

    pub fn voxel_index_to_world(&self, index: &VoxelIndex) -> Vector3<f32> {
        let x = self.config.min_bound.x + (index.x as f32 + 0.5) * self.config.voxel_size;
        let y = self.config.min_bound.y + (index.y as f32 + 0.5) * self.config.voxel_size;
        let z = self.config.min_bound.z + (index.z as f32 + 0.5) * self.config.voxel_size;
        Vector3::new(x, y, z)
    }

    pub fn is_within_bounds(&self, position: &Vector3<f32>) -> bool {
        position.x >= self.config.min_bound.x
            && position.x <= self.config.max_bound.x
            && position.y >= self.config.min_bound.y
            && position.y <= self.config.max_bound.y
            && position.z >= self.config.min_bound.z
            && position.z <= self.config.max_bound.z
    }

    pub fn get_voxel(&self, index: &VoxelIndex) -> Option<&Voxel> {
        self.voxels.get(index)
    }

    pub fn get_or_create_voxel(&mut self, index: VoxelIndex) -> &mut Voxel {
        let center = self.voxel_index_to_world(&index);
        self.voxels
            .entry(index)
            .or_insert_with(|| Voxel::new(center))
    }

    pub fn occupied_voxel_count(&self) -> usize {
        self.voxels
            .values()
            .filter(|voxel| voxel.state == VoxelState::Occupied)
            .count()
    }

    pub fn free_voxel_count(&self) -> usize {
        self.voxels
            .values()
            .filter(|voxel| voxel.state == VoxelState::Free)
            .count()
    }

    pub fn unknown_voxel_count(&self) -> usize {
        self.voxels
            .values()
            .filter(|voxel| voxel.state == VoxelState::Unknown)
            .count()
    }
}
