use anyhow::{Context, Result};
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub log_level: tracing::Level,
    pub nats_url: String,
    pub nats_stream_name: String,
    pub subject_filter: String,
    pub postgres_url: String,
    pub postgres_max_connection_count: u8,
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
            log_level: std::env::var("LOG_LEVEL")
                .unwrap_or_else(|_| "INFO".to_string())
                .parse::<tracing::Level>()
                .context(
                    "LOG_LEVEL must be a valid tracing level (TRACE, DEBUG, INFO, WARN, ERROR)",
                )?,
            nats_url: std::env::var("NATS_URL").context("NATS_URL must be set")?,
            nats_stream_name: std::env::var("NATS_STREAM_NAME")
                .context("NATS_STREAM_NAME must be set")?,
            subject_filter: std::env::var("SUBJECT_FILTER")
                .context("SUBJECT_FILTER must be set")?,
            postgres_url: std::env::var("POSTGRES_URL").context("POSTGRES_URL must be set")?,
            postgres_max_connection_count: std::env::var("POSTGRES_MAX_CONNECTION_COUNT")
                .context("POSTGRES_MAX_CONNECTION_COUNT must be set")?
                .parse()
                .context("POSTGRES_MAX_CONNECTION_COUNT must be a valid positive integer")?,
        };
        Ok(app_config)
    }

    pub fn get() -> &'static AppConfig {
        static CONFIG: OnceLock<AppConfig> = OnceLock::new();
        CONFIG.get_or_init(|| AppConfig::load().expect("Failed to load application configuration"))
    }
}
