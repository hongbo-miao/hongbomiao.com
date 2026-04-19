use anyhow::{Context, Result};

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub pulsar_url: String,
    pub pulsar_topic: String,
    pub pulsar_output_topic: String,
    pub livekit_url: String,
    pub livekit_api_key: String,
    pub livekit_api_secret: String,
    pub livekit_room: String,
    pub instance_id: String,
    pub asr_model_dir: String,
    pub silero_vad_model_dir: String,
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
            pulsar_output_topic: std::env::var("PULSAR_OUTPUT_TOPIC")
                .context("PULSAR_OUTPUT_TOPIC must be set")?,
            livekit_url: std::env::var("LIVEKIT_URL").context("LIVEKIT_URL must be set")?,
            livekit_api_key: std::env::var("LIVEKIT_API_KEY")
                .context("LIVEKIT_API_KEY must be set")?,
            livekit_api_secret: std::env::var("LIVEKIT_API_SECRET")
                .context("LIVEKIT_API_SECRET must be set")?,
            livekit_room: std::env::var("LIVEKIT_ROOM").context("LIVEKIT_ROOM must be set")?,
            instance_id: std::env::var("HOSTNAME").context("HOSTNAME must be set")?,
            asr_model_dir: std::env::var("ASR_MODEL_DIR").context("ASR_MODEL_DIR must be set")?,
            silero_vad_model_dir: std::env::var("SILERO_VAD_MODEL_DIR")
                .context("SILERO_VAD_MODEL_DIR must be set")?,
        };
        Ok(app_config)
    }
}
