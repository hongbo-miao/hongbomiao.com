use anyhow::{Context, Result};
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub movement_velocity_threshold_mps: f64,
    pub camera_confidence_weight: f64,
    pub fusion_base_confidence: f64,
    pub max_frame_count: usize,
    pub yolo_model_path: String,
    pub association_distance_threshold_pixels: f64,
    pub occupancy_voxel_size_m: f32,
    pub occupancy_min_bound_x_m: f32,
    pub occupancy_min_bound_y_m: f32,
    pub occupancy_min_bound_z_m: f32,
    pub occupancy_max_bound_x_m: f32,
    pub occupancy_max_bound_y_m: f32,
    pub occupancy_max_bound_z_m: f32,
    pub occupancy_decay_rate: f32,
    pub occupancy_clear_distance_m: f32,
    pub occupancy_occupied_threshold: f32,
    pub occupancy_free_threshold: f32,
    pub occupancy_occupied_probability_increment: f32,
    pub occupancy_free_probability_decrement: f32,
}

impl AppConfig {
    pub fn load() -> Result<Self> {
        if cfg!(test) {
            if let Err(error) = dotenvy::from_filename(".env.development") {
                eprintln!("Failed to load .env.development: {error}");
            }
            let _ = dotenvy::from_filename_override(".env.development.local");
        } else {
            let environment = std::env::var("ENVIRONMENT").context("ENVIRONMENT must be set")?;

            match environment.as_str() {
                "development" | "production" => {
                    let env_file = format!(".env.{environment}");
                    let env_local_file = format!(".env.{environment}.local");

                    dotenvy::from_filename(&env_file)
                        .context(format!("Failed to load {env_file} file"))?;
                    let _ = dotenvy::from_filename_override(&env_local_file);
                }
                environment => {
                    return Err(anyhow::anyhow!(
                        "Unknown ENVIRONMENT value '{environment}'."
                    ));
                }
            }
        }

        let app_config = AppConfig {
            movement_velocity_threshold_mps: std::env::var("MOVEMENT_VELOCITY_THRESHOLD_MPS")
                .context("MOVEMENT_VELOCITY_THRESHOLD_MPS must be set")?
                .parse::<f64>()
                .context("MOVEMENT_VELOCITY_THRESHOLD_MPS must be a valid number")?,
            camera_confidence_weight: std::env::var("CAMERA_CONFIDENCE_WEIGHT")
                .context("CAMERA_CONFIDENCE_WEIGHT must be set")?
                .parse::<f64>()
                .context("CAMERA_CONFIDENCE_WEIGHT must be a valid number")?,
            fusion_base_confidence: std::env::var("FUSION_BASE_CONFIDENCE")
                .context("FUSION_BASE_CONFIDENCE must be set")?
                .parse::<f64>()
                .context("FUSION_BASE_CONFIDENCE must be a valid number")?,
            max_frame_count: std::env::var("MAX_FRAME_COUNT")
                .context("MAX_FRAME_COUNT must be set")?
                .parse::<usize>()
                .context("MAX_FRAME_COUNT must be a valid number")?,
            yolo_model_path: std::env::var("YOLO_MODEL_PATH")
                .context("YOLO_MODEL_PATH must be set")?,
            association_distance_threshold_pixels: std::env::var(
                "ASSOCIATION_DISTANCE_THRESHOLD_PIXELS",
            )
            .context("ASSOCIATION_DISTANCE_THRESHOLD_PIXELS must be set")?
            .parse::<f64>()
            .context("ASSOCIATION_DISTANCE_THRESHOLD_PIXELS must be a valid number")?,
            occupancy_voxel_size_m: std::env::var("OCCUPANCY_VOXEL_SIZE_M")
                .context("OCCUPANCY_VOXEL_SIZE_M must be set")?
                .parse::<f32>()
                .context("OCCUPANCY_VOXEL_SIZE_M must be a valid number")?,
            occupancy_min_bound_x_m: std::env::var("OCCUPANCY_MIN_BOUND_X_M")
                .context("OCCUPANCY_MIN_BOUND_X_M must be set")?
                .parse::<f32>()
                .context("OCCUPANCY_MIN_BOUND_X_M must be a valid number")?,
            occupancy_min_bound_y_m: std::env::var("OCCUPANCY_MIN_BOUND_Y_M")
                .context("OCCUPANCY_MIN_BOUND_Y_M must be set")?
                .parse::<f32>()
                .context("OCCUPANCY_MIN_BOUND_Y_M must be a valid number")?,
            occupancy_min_bound_z_m: std::env::var("OCCUPANCY_MIN_BOUND_Z_M")
                .context("OCCUPANCY_MIN_BOUND_Z_M must be set")?
                .parse::<f32>()
                .context("OCCUPANCY_MIN_BOUND_Z_M must be a valid number")?,
            occupancy_max_bound_x_m: std::env::var("OCCUPANCY_MAX_BOUND_X_M")
                .context("OCCUPANCY_MAX_BOUND_X_M must be set")?
                .parse::<f32>()
                .context("OCCUPANCY_MAX_BOUND_X_M must be a valid number")?,
            occupancy_max_bound_y_m: std::env::var("OCCUPANCY_MAX_BOUND_Y_M")
                .context("OCCUPANCY_MAX_BOUND_Y_M must be set")?
                .parse::<f32>()
                .context("OCCUPANCY_MAX_BOUND_Y_M must be a valid number")?,
            occupancy_max_bound_z_m: std::env::var("OCCUPANCY_MAX_BOUND_Z_M")
                .context("OCCUPANCY_MAX_BOUND_Z_M must be set")?
                .parse::<f32>()
                .context("OCCUPANCY_MAX_BOUND_Z_M must be a valid number")?,
            occupancy_decay_rate: std::env::var("OCCUPANCY_DECAY_RATE")
                .context("OCCUPANCY_DECAY_RATE must be set")?
                .parse::<f32>()
                .context("OCCUPANCY_DECAY_RATE must be a valid number")?,
            occupancy_clear_distance_m: std::env::var("OCCUPANCY_CLEAR_DISTANCE_M")
                .context("OCCUPANCY_CLEAR_DISTANCE_M must be set")?
                .parse::<f32>()
                .context("OCCUPANCY_CLEAR_DISTANCE_M must be a valid number")?,
            occupancy_occupied_threshold: std::env::var("OCCUPANCY_OCCUPIED_THRESHOLD")
                .context("OCCUPANCY_OCCUPIED_THRESHOLD must be set")?
                .parse::<f32>()
                .context("OCCUPANCY_OCCUPIED_THRESHOLD must be a valid number")?,
            occupancy_free_threshold: std::env::var("OCCUPANCY_FREE_THRESHOLD")
                .context("OCCUPANCY_FREE_THRESHOLD must be set")?
                .parse::<f32>()
                .context("OCCUPANCY_FREE_THRESHOLD must be a valid number")?,
            occupancy_occupied_probability_increment: std::env::var(
                "OCCUPANCY_OCCUPIED_PROBABILITY_INCREMENT",
            )
            .context("OCCUPANCY_OCCUPIED_PROBABILITY_INCREMENT must be set")?
            .parse::<f32>()
            .context("OCCUPANCY_OCCUPIED_PROBABILITY_INCREMENT must be a valid number")?,
            occupancy_free_probability_decrement: std::env::var(
                "OCCUPANCY_FREE_PROBABILITY_DECREMENT",
            )
            .context("OCCUPANCY_FREE_PROBABILITY_DECREMENT must be set")?
            .parse::<f32>()
            .context("OCCUPANCY_FREE_PROBABILITY_DECREMENT must be a valid number")?,
        };
        Ok(app_config)
    }

    pub fn get() -> &'static AppConfig {
        static CONFIG: OnceLock<AppConfig> = OnceLock::new();
        CONFIG.get_or_init(|| AppConfig::load().expect("Failed to load application configuration"))
    }
}
