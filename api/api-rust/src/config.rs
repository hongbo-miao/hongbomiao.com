use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub server_port: u16,
    pub server_log_level: tracing::Level,
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
    pub webrtc_vad_sample_rate_number: u32,
    pub webrtc_vad_frame_duration_ms: u32,
    pub webrtc_vad_mode: String,
    pub webrtc_vad_debounce_frame_number: usize,
    pub webrtc_vad_min_speech_duration_ms: u32,
    pub webrtc_vad_min_silence_duration_ms: u32,
    pub webrtc_vad_padding_duration_ms: u32,
}

impl AppConfig {
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        if cfg!(test) {
            let _ = dotenvy::from_filename(".env.development");
            let _ = dotenvy::from_filename_override(".env.development.local");
        } else {
            match std::env::var("RUST_ENV")
                .unwrap_or_else(|_| "development".to_string())
                .as_str()
            {
                "development" => {
                    let _ = dotenvy::from_filename(".env.development");
                    let _ = dotenvy::from_filename_override(".env.development.local");
                }
                "production" => {
                    let _ = dotenvy::from_filename(".env.production");
                    let _ = dotenvy::from_filename_override(".env.production.local");
                }
                env => {
                    eprintln!("Warning: Unknown RUST_ENV value '{env}', defaulting to no env file");
                }
            }
        }

        let app_config = AppConfig {
            server_port: std::env::var("SERVER_PORT")?.parse()?,
            server_log_level: std::env::var("SERVER_LOG_LEVEL")
                .unwrap_or_else(|_| "INFO".to_string())
                .parse::<tracing::Level>()
                .expect(
                    "SERVER_LOG_LEVEL must be a valid tracing level (TRACE, DEBUG, INFO, WARN, ERROR)",
                ),
            openai_api_base_url: std::env::var("OPENAI_API_BASE_URL")?,
            openai_api_key: std::env::var("OPENAI_API_KEY")?,
            openai_model: std::env::var("OPENAI_MODEL")?,
            speaches_base_url: std::env::var("SPEACHES_BASE_URL")?,
            transcription_model: std::env::var("TRANSCRIPTION_MODEL")?,
            server_cors_allowed_origins: std::env::var("SERVER_CORS_ALLOWED_ORIGINS")?
                .split(',')
                .map(|origin| origin.trim().to_string())
                .filter(|origin| !origin.is_empty())
                .collect(),
            server_rate_limit_per_second: std::env::var("SERVER_RATE_LIMIT_PER_SECOND")?.parse()?,
            server_rate_limit_per_second_burst: std::env::var("SERVER_RATE_LIMIT_PER_SECOND_BURST")?.parse()?,
            server_sent_event_retry_ms: std::env::var(
                "SERVER_SENT_EVENT_RETRY_MS",
            )?
            .parse()?,
            server_sent_event_keep_alive_interval_s: std::env::var(
                "SERVER_SENT_EVENT_KEEP_ALIVE_INTERVAL_S",
            )?
            .parse()?,
            webrtc_vad_min_silence_duration_ms: std::env::var(
                "WEBRTC_VAD_MIN_SILENCE_DURATION_MS",
            )?
            .parse()?,
            webrtc_vad_min_speech_duration_ms: std::env::var("WEBRTC_VAD_MIN_SPEECH_DURATION_MS")?
                .parse()?,
            webrtc_vad_padding_duration_ms: std::env::var("WEBRTC_VAD_PADDING_DURATION_MS")?
                .parse()?,
            webrtc_vad_frame_duration_ms: std::env::var("WEBRTC_VAD_FRAME_DURATION_MS")?.parse()?,
            webrtc_vad_sample_rate_number: std::env::var("WEBRTC_VAD_SAMPLE_RATE_NUMBER")?
                .parse()?,
            webrtc_vad_mode: std::env::var("WEBRTC_VAD_MODE")?,
            webrtc_vad_debounce_frame_number: std::env::var("WEBRTC_VAD_DEBOUNCE_FRAME_NUMBER")?
                .parse()?,
        };
        Ok(app_config)
    }

    pub fn get() -> &'static AppConfig {
        static CONFIG: OnceLock<AppConfig> = OnceLock::new();
        CONFIG.get_or_init(|| AppConfig::load().expect("Failed to load application configuration"))
    }
}
