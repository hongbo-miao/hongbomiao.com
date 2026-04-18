use anyhow::{Context, Result};
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub livekit_url: String,
    pub livekit_api_key: String,
    pub livekit_api_secret: String,
    pub livekit_room: String,
    pub pulsar_url: String,
    pub pulsar_topic: String,
    pub hostname: String,
    pub http_port: u16,
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
            livekit_url: std::env::var("LIVEKIT_URL").context("LIVEKIT_URL must be set")?,
            livekit_api_key: std::env::var("LIVEKIT_API_KEY")
                .context("LIVEKIT_API_KEY must be set")?,
            livekit_api_secret: std::env::var("LIVEKIT_API_SECRET")
                .context("LIVEKIT_API_SECRET must be set")?,
            livekit_room: std::env::var("LIVEKIT_ROOM").context("LIVEKIT_ROOM must be set")?,
            pulsar_url: std::env::var("PULSAR_URL").context("PULSAR_URL must be set")?,
            pulsar_topic: std::env::var("PULSAR_TOPIC").context("PULSAR_TOPIC must be set")?,
            hostname: std::env::var("HOSTNAME").context("HOSTNAME must be set")?,
            http_port: std::env::var("HTTP_PORT")
                .context("HTTP_PORT must be set")?
                .parse()
                .context("HTTP_PORT must be a valid port number")?,
        };
        Ok(app_config)
    }

    pub fn get() -> &'static AppConfig {
        static CONFIG: OnceLock<AppConfig> = OnceLock::new();
        CONFIG.get_or_init(|| AppConfig::load().expect("Failed to load application configuration"))
    }
}
