use std::sync::OnceLock;

use anyhow::{Context, Result};
use rand::prelude::IndexedRandom;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub ingest_url: String,
    pub device_id: String,
    pub wav_path: String,
}

fn pick_random_wav_file(data_dir: &str) -> Result<String> {
    let wav_paths: Vec<_> = std::fs::read_dir(data_dir)
        .with_context(|| format!("Failed to read data directory '{data_dir}'"))?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension()?.eq_ignore_ascii_case("wav") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    wav_paths
        .choose(&mut rand::rng())
        .map(|path| path.to_string_lossy().into_owned())
        .context("No WAV files found in data directory")
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
            wav_path: pick_random_wav_file("data")?,
        };
        Ok(app_config)
    }

    pub fn get() -> &'static AppConfig {
        static CONFIG: OnceLock<AppConfig> = OnceLock::new();
        CONFIG.get_or_init(|| AppConfig::load().expect("Failed to load application configuration"))
    }
}
