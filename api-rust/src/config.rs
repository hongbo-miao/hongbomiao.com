use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub port: u16,
    pub log_level: tracing::Level,
    pub openai_api_base_url: String,
    pub openai_api_key: String,
    pub openai_model: String,
    pub speaches_base_url: String,
    pub transcription_model: String,
    pub cors_allowed_origins: Vec<String>,
    pub webrtc_vad_sample_rate_number: u32,
    pub webrtc_vad_frame_duration_ms: u64,
    pub webrtc_vad_mode: String,
    pub webrtc_vad_debounce_frame_number: usize,
    pub webrtc_vad_min_speech_duration_ms: u64,
    pub webrtc_vad_min_silence_duration_ms: u64,
    pub webrtc_vad_padding_duration_ms: u64,
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
            port: std::env::var("PORT")?.parse()?,
            log_level: std::env::var("LOG_LEVEL")
                .unwrap_or_else(|_| "INFO".to_string())
                .parse::<tracing::Level>()
                .expect(
                    "LOG_LEVEL must be a valid tracing level (TRACE, DEBUG, INFO, WARN, ERROR)",
                ),
            openai_api_base_url: std::env::var("OPENAI_API_BASE_URL")?,
            openai_api_key: std::env::var("OPENAI_API_KEY")?,
            openai_model: std::env::var("OPENAI_MODEL")?,
            speaches_base_url: std::env::var("SPEACHES_BASE_URL")?,
            transcription_model: std::env::var("TRANSCRIPTION_MODEL")?,
            cors_allowed_origins: std::env::var("CORS_ALLOWED_ORIGINS")
                .unwrap_or_else(|_| "*".to_string())
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
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
