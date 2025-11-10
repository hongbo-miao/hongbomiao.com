use anyhow::{Context, Result};
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub log_level: tracing::Level,
    pub nats_url: String,
    pub nats_stream_name: String,
    pub subject_filter: String,
    pub speaches_base_url: String,
    pub transcription_model: String,
    pub webrtc_vad_debounce_frame_number: usize,
    pub webrtc_vad_frame_duration_ms: u32,
    pub webrtc_vad_min_silence_duration_ms: u32,
    pub webrtc_vad_min_speech_duration_ms: u32,
    pub webrtc_vad_mode: String,
    pub webrtc_vad_padding_duration_ms: u32,
    pub webrtc_vad_sample_rate_number: u32,
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
            speaches_base_url: std::env::var("SPEACHES_BASE_URL")
                .context("SPEACHES_BASE_URL must be set")?,
            transcription_model: std::env::var("TRANSCRIPTION_MODEL")
                .context("TRANSCRIPTION_MODEL must be set")?,
            webrtc_vad_debounce_frame_number: std::env::var("WEBRTC_VAD_DEBOUNCE_FRAME_NUMBER")
                .context("WEBRTC_VAD_DEBOUNCE_FRAME_NUMBER must be set")?
                .parse::<usize>()
                .context("WEBRTC_VAD_DEBOUNCE_FRAME_NUMBER must be a valid positive integer")?,
            webrtc_vad_frame_duration_ms: std::env::var("WEBRTC_VAD_FRAME_DURATION_MS")
                .context("WEBRTC_VAD_FRAME_DURATION_MS must be set")?
                .parse::<u32>()
                .context("WEBRTC_VAD_FRAME_DURATION_MS must be a valid positive integer")?,
            webrtc_vad_min_silence_duration_ms: std::env::var("WEBRTC_VAD_MIN_SILENCE_DURATION_MS")
                .context("WEBRTC_VAD_MIN_SILENCE_DURATION_MS must be set")?
                .parse::<u32>()
                .context("WEBRTC_VAD_MIN_SILENCE_DURATION_MS must be a valid positive integer")?,
            webrtc_vad_min_speech_duration_ms: std::env::var("WEBRTC_VAD_MIN_SPEECH_DURATION_MS")
                .context("WEBRTC_VAD_MIN_SPEECH_DURATION_MS must be set")?
                .parse::<u32>()
                .context("WEBRTC_VAD_MIN_SPEECH_DURATION_MS must be a valid positive integer")?,
            webrtc_vad_mode: std::env::var("WEBRTC_VAD_MODE")
                .context("WEBRTC_VAD_MODE must be set")?,
            webrtc_vad_padding_duration_ms: std::env::var("WEBRTC_VAD_PADDING_DURATION_MS")
                .context("WEBRTC_VAD_PADDING_DURATION_MS must be set")?
                .parse::<u32>()
                .context("WEBRTC_VAD_PADDING_DURATION_MS must be a valid positive integer")?,
            webrtc_vad_sample_rate_number: std::env::var("WEBRTC_VAD_SAMPLE_RATE_NUMBER")
                .context("WEBRTC_VAD_SAMPLE_RATE_NUMBER must be set")?
                .parse::<u32>()
                .context("WEBRTC_VAD_SAMPLE_RATE_NUMBER must be a valid positive integer")?,
        };
        Ok(app_config)
    }

    pub fn get() -> &'static AppConfig {
        static CONFIG: OnceLock<AppConfig> = OnceLock::new();
        CONFIG.get_or_init(|| AppConfig::load().expect("Failed to load application configuration"))
    }
}
