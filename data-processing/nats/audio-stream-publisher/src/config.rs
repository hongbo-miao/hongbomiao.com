use anyhow::{Context, Result};
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub log_level: tracing::Level,
    pub nats_url: String,
    pub fire_stream_name: String,
    pub fire_stream_identifier: String,
    pub fire_stream_url: String,
    pub fire_stream_location: String,
    pub subject_prefix: String,
    pub pcm_chunk_size_bytes: u32,
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
                    eprintln!("Unknown ENVIRONMENT value '{environment}'");
                }
            }
        }

        let app_config = AppConfig {
            log_level: std::env::var("LOG_LEVEL")
                .unwrap_or_else(|_| "INFO".to_string())
                .parse::<tracing::Level>()
                .context(
                    "LOG_LEVEL must be a valid tracing level (TRACE, DEBUG, INFO, WARN, ERROR)",
                )?,
            nats_url: std::env::var("NATS_URL").context("NATS_URL must be set")?,
            fire_stream_name: std::env::var("FIRE_STREAM_NAME")
                .context("FIRE_STREAM_NAME must be set")?,
            fire_stream_identifier: std::env::var("FIRE_STREAM_IDENTIFIER")
                .context("FIRE_STREAM_IDENTIFIER must be set")?,
            fire_stream_url: std::env::var("FIRE_STREAM_URL")
                .context("FIRE_STREAM_URL must be set")?,
            fire_stream_location: std::env::var("FIRE_STREAM_LOCATION")
                .context("FIRE_STREAM_LOCATION must be set")?,
            subject_prefix: std::env::var("SUBJECT_PREFIX")
                .context("SUBJECT_PREFIX must be set")?,
            pcm_chunk_size_bytes: std::env::var("PCM_CHUNK_SIZE_BYTES")
                .context("PCM_CHUNK_SIZE_BYTES must be set")?
                .parse::<u32>()
                .context("PCM_CHUNK_SIZE_BYTES must be a valid positive integer")?,
        };
        Ok(app_config)
    }

    pub fn get() -> &'static AppConfig {
        static CONFIG: OnceLock<AppConfig> = OnceLock::new();
        CONFIG.get_or_init(|| AppConfig::load().expect("Failed to load application configuration"))
    }
}
