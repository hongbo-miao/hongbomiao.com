use std::sync::OnceLock;

use anyhow::{Context, Result};

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub pulsar_url: String,
    pub pulsar_topic: String,
    pub grpc_port: u16,
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
            pulsar_url: std::env::var("PULSAR_URL").context("PULSAR_URL must be set")?,
            pulsar_topic: std::env::var("PULSAR_TOPIC").context("PULSAR_TOPIC must be set")?,
            grpc_port: std::env::var("GRPC_PORT")
                .context("GRPC_PORT must be set")?
                .parse()
                .context("GRPC_PORT must be a valid port number")?,
        };
        Ok(app_config)
    }

    pub fn get() -> &'static AppConfig {
        static CONFIG: OnceLock<AppConfig> = OnceLock::new();
        CONFIG.get_or_init(|| AppConfig::load().expect("Failed to load application configuration"))
    }
}
