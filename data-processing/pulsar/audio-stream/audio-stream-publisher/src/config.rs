use std::sync::OnceLock;

use anyhow::{Context, Result};

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub ingest_url: String,
    pub device_id: String,
    pub wav_path: String,
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

                    let _ = dotenvy::from_filename(&env_file);
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
            ingest_url: std::env::var("INGEST_URL").context("INGEST_URL must be set")?,
            device_id: std::env::var("DEVICE_ID")
                .unwrap_or_else(|_| format!("device-{}", &uuid::Uuid::new_v4().to_string()[..8])),
            wav_path: std::env::var("WAV_PATH").context("WAV_PATH must be set")?,
        };
        Ok(app_config)
    }

    pub fn get() -> &'static AppConfig {
        static CONFIG: OnceLock<AppConfig> = OnceLock::new();
        CONFIG.get_or_init(|| AppConfig::load().expect("Failed to load application configuration"))
    }
}
