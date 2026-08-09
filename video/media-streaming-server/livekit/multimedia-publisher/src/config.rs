use anyhow::{Context, Result};
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub livekit_url: String,
    pub livekit_api_key: String,
    pub livekit_api_secret: String,
    pub livekit_room: String,
    pub http_port: u16,
    pub s3_endpoint: String,
    pub s3_region: String,
    pub s3_access_key: String,
    pub s3_secret_key: String,
    pub hls_bucket: String,
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
            http_port: std::env::var("HTTP_PORT")
                .context("HTTP_PORT must be set")?
                .parse()
                .context("HTTP_PORT must be a valid port number")?,
            s3_endpoint: std::env::var("S3_ENDPOINT").context("S3_ENDPOINT must be set")?,
            s3_region: std::env::var("S3_REGION").context("S3_REGION must be set")?,
            s3_access_key: std::env::var("S3_ACCESS_KEY").context("S3_ACCESS_KEY must be set")?,
            s3_secret_key: std::env::var("S3_SECRET_KEY").context("S3_SECRET_KEY must be set")?,
            hls_bucket: std::env::var("HLS_BUCKET").context("HLS_BUCKET must be set")?,
        };
        Ok(app_config)
    }

    pub fn get() -> &'static AppConfig {
        static CONFIG: OnceLock<AppConfig> = OnceLock::new();
        CONFIG.get_or_init(|| AppConfig::load().expect("Failed to load application configuration"))
    }
}
