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
    pub occupancy_log_odds: f32,
    pub center: Vector3<f32>,
}

impl Voxel {
    pub fn new(center: Vector3<f32>) -> Self {
        Self {
            state: VoxelState::Unknown,
            occupancy_probability: 0.5,
            occupancy_log_odds: 0.0,
            center,
        }
    }

    // Bayesian occupancy grids with log-odds updates
    pub fn update_occupancy(
        &mut self,
        is_occupied: bool,
        occupied_threshold: f32,
        free_threshold: f32,
        occupied_probability_given_occupied_evidence: f32,
        occupied_probability_given_free_evidence: f32,
    ) {
        let measurement_probability = if is_occupied {
            occupied_probability_given_occupied_evidence
        } else {
            occupied_probability_given_free_evidence
        };

        // Clamp away from 0 and 1 to avoid infinite log-odds and keep updates numerically stable
        // ln(0.001 / 0.999) ≈ -6.9 and ln(0.999 / 0.001) ≈ +6.9, so each update is strong but finite
        let clamped_measurement_probability = measurement_probability.clamp(0.001, 0.999);
        let measurement_log_odds =
            (clamped_measurement_probability / (1.0 - clamped_measurement_probability)).ln();
        self.occupancy_log_odds += measurement_log_odds;
        self.occupancy_probability = 1.0 / (1.0 + (-self.occupancy_log_odds).exp());

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
    pub occupied_probability_given_occupied_evidence: f32,
    pub occupied_probability_given_free_evidence: f32,
}

impl OccupancyGridConfig {
    pub fn new(
        voxel_size: f32,
        min_bound: Vector3<f32>,
        max_bound: Vector3<f32>,
        occupied_threshold: f32,
        free_threshold: f32,
        occupied_probability_given_occupied_evidence: f32,
        occupied_probability_given_free_evidence: f32,
    ) -> Self {
        Self {
            voxel_size,
            min_bound,
            max_bound,
            occupied_threshold,
            free_threshold,
            occupied_probability_given_occupied_evidence,
            occupied_probability_given_free_evidence,
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
