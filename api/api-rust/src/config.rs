use anyhow::{Context, Result};
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub server_port: u16,
    pub server_log_level: tracing::Level,
    pub database_url: String,
    pub database_max_connection_count: u8,
    pub openai_api_base_url: String,
    pub openai_api_key: String,
    pub openai_model: String,
    pub speaches_base_url: String,
    pub transcription_model: String,
    pub server_cors_allowed_origins: Vec<String>,
    pub server_rate_limit_per_second: u16,
    pub server_rate_limit_per_second_burst: u16,
    pub server_sent_event_retry_ms: u16,
    pub server_sent_event_keep_alive_interval_s: u16,
    pub webtransport_certificate_path: String,
    pub webtransport_private_key_path: String,
    pub webrtc_vad_sample_rate_number: u32,
    pub webrtc_vad_frame_duration_ms: u32,
    pub webrtc_vad_mode: String,
    pub webrtc_vad_debounce_frame_number: usize,
    pub webrtc_vad_min_speech_duration_ms: u32,
    pub webrtc_vad_min_silence_duration_ms: u32,
    pub webrtc_vad_padding_duration_ms: u32,
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
            server_port: std::env::var("SERVER_PORT")
                .context("SERVER_PORT must be set")?
                .parse()
                .context("SERVER_PORT must be a valid port number")?,
            server_log_level: std::env::var("SERVER_LOG_LEVEL")
                .unwrap_or_else(|_| "INFO".to_string())
                .parse::<tracing::Level>()
                .context(
                    "SERVER_LOG_LEVEL must be a valid tracing level (TRACE, DEBUG, INFO, WARN, ERROR)",
                )?,
            database_url: std::env::var("DATABASE_URL")
                .context("DATABASE_URL must be set")?,
            database_max_connection_count: std::env::var("DATABASE_MAX_CONNECTION_COUNT")
                .context("DATABASE_MAX_CONNECTION_COUNT must be set")?
                .parse()
                .context("DATABASE_MAX_CONNECTION_COUNT must be a valid positive integer")?,
            openai_api_base_url: std::env::var("OPENAI_API_BASE_URL")
                .context("OPENAI_API_BASE_URL must be set")?,
            openai_api_key: std::env::var("OPENAI_API_KEY")
                .context("OPENAI_API_KEY must be set")?,
            openai_model: std::env::var("OPENAI_MODEL")
                .context("OPENAI_MODEL must be set")?,
            speaches_base_url: std::env::var("SPEACHES_BASE_URL")
                .context("SPEACHES_BASE_URL must be set")?,
            transcription_model: std::env::var("TRANSCRIPTION_MODEL")
                .context("TRANSCRIPTION_MODEL must be set")?,
            server_cors_allowed_origins: std::env::var("SERVER_CORS_ALLOWED_ORIGINS")
                .context("SERVER_CORS_ALLOWED_ORIGINS must be set")?
                .split(',')
                .map(|origin| origin.trim().to_string())
                .filter(|origin| !origin.is_empty())
                .collect(),
            server_rate_limit_per_second: std::env::var("SERVER_RATE_LIMIT_PER_SECOND")
                .context("SERVER_RATE_LIMIT_PER_SECOND must be set")?
                .parse()
                .context("SERVER_RATE_LIMIT_PER_SECOND must be a valid positive integer")?,
            server_rate_limit_per_second_burst: std::env::var("SERVER_RATE_LIMIT_PER_SECOND_BURST")
                .context("SERVER_RATE_LIMIT_PER_SECOND_BURST must be set")?
                .parse()
                .context("SERVER_RATE_LIMIT_PER_SECOND_BURST must be a valid positive integer")?,
            server_sent_event_retry_ms: std::env::var("SERVER_SENT_EVENT_RETRY_MS")
                .context("SERVER_SENT_EVENT_RETRY_MS must be set")?
                .parse()
                .context("SERVER_SENT_EVENT_RETRY_MS must be a valid positive integer")?,
            server_sent_event_keep_alive_interval_s: std::env::var("SERVER_SENT_EVENT_KEEP_ALIVE_INTERVAL_S")
                .context("SERVER_SENT_EVENT_KEEP_ALIVE_INTERVAL_S must be set")?
                .parse()
                .context("SERVER_SENT_EVENT_KEEP_ALIVE_INTERVAL_S must be a valid positive integer")?,
            webtransport_certificate_path: std::env::var("WEBTRANSPORT_CERTIFICATE_PATH")
                .context("WEBTRANSPORT_CERTIFICATE_PATH must be set")?,
            webtransport_private_key_path: std::env::var("WEBTRANSPORT_PRIVATE_KEY_PATH")
                .context("WEBTRANSPORT_PRIVATE_KEY_PATH must be set")?,
            webrtc_vad_min_silence_duration_ms: std::env::var("WEBRTC_VAD_MIN_SILENCE_DURATION_MS")
                .context("WEBRTC_VAD_MIN_SILENCE_DURATION_MS must be set")?
                .parse()
                .context("WEBRTC_VAD_MIN_SILENCE_DURATION_MS must be a valid positive integer")?,
            webrtc_vad_min_speech_duration_ms: std::env::var("WEBRTC_VAD_MIN_SPEECH_DURATION_MS")
                .context("WEBRTC_VAD_MIN_SPEECH_DURATION_MS must be set")?
                .parse()
                .context("WEBRTC_VAD_MIN_SPEECH_DURATION_MS must be a valid positive integer")?,
            webrtc_vad_padding_duration_ms: std::env::var("WEBRTC_VAD_PADDING_DURATION_MS")
                .context("WEBRTC_VAD_PADDING_DURATION_MS must be set")?
                .parse()
                .context("WEBRTC_VAD_PADDING_DURATION_MS must be a valid positive integer")?,
            webrtc_vad_frame_duration_ms: std::env::var("WEBRTC_VAD_FRAME_DURATION_MS")
                .context("WEBRTC_VAD_FRAME_DURATION_MS must be set")?
                .parse()
                .context("WEBRTC_VAD_FRAME_DURATION_MS must be a valid positive integer")?,
            webrtc_vad_sample_rate_number: std::env::var("WEBRTC_VAD_SAMPLE_RATE_NUMBER")
                .context("WEBRTC_VAD_SAMPLE_RATE_NUMBER must be set")?
                .parse()
                .context("WEBRTC_VAD_SAMPLE_RATE_NUMBER must be a valid positive integer")?,
            webrtc_vad_mode: std::env::var("WEBRTC_VAD_MODE")
                .context("WEBRTC_VAD_MODE must be set")?,
            webrtc_vad_debounce_frame_number: std::env::var("WEBRTC_VAD_DEBOUNCE_FRAME_NUMBER")
                .context("WEBRTC_VAD_DEBOUNCE_FRAME_NUMBER must be set")?
                .parse()
                .context("WEBRTC_VAD_DEBOUNCE_FRAME_NUMBER must be a valid positive integer")?,
        };
        Ok(app_config)
    }

    pub fn get() -> &'static AppConfig {
        static CONFIG: OnceLock<AppConfig> = OnceLock::new();
        CONFIG.get_or_init(|| AppConfig::load().expect("Failed to load application configuration"))
    }
}
