use anyhow::{Context, Result};
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub movement_velocity_threshold_mps: f64,
    pub camera_confidence_weight: f64,
    pub fusion_base_confidence: f64,
    pub max_frame_count: usize,
    pub yolo_model_path: String,
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
        };
        Ok(app_config)
    }

    pub fn get() -> &'static AppConfig {
        static CONFIG: OnceLock<AppConfig> = OnceLock::new();
        CONFIG.get_or_init(|| AppConfig::load().expect("Failed to load application configuration"))
    }
}
