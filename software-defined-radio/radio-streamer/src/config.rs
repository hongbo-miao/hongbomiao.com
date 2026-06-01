//! Configuration from environment variables (.env files). `dotenvy` loads
//! `.env.{ENVIRONMENT}` then overrides with `.env.{ENVIRONMENT}.local`, and
//! every value is read from an env var.

use std::sync::OnceLock;

use anyhow::{Context, Result};

/// Half the bandwidth each channel occupies around its centre, in Hz. Matches
/// the DDC low-pass cutoff (`DDC_LOWPASS_HZ` in `channel`), used to keep every
/// channel clear of the captured spectrum edges.
const CHANNEL_HALF_BANDWIDTH_HZ: i64 = 100_000;

/// One configured channel. Parsed from the `CHANNELS` env var, where each entry
/// is `name|frequency_hz` and entries are comma-separated.
#[derive(Debug, Clone)]
pub struct Channel {
    pub name: String,
    pub freq: u64,
}

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub sdr_backend: String,
    /// SoapySDR device args (e.g. `"driver=airspy"`).
    pub sdr_device_args: String,
    pub sdr_sample_rate: u32,
    pub sdr_bandwidth_hz: Option<u32>,
    /// Derived from the channel frequencies (midpoint shifted by lo_offset). Not an env var.
    pub sdr_center_freq: u64,
    /// `None` selects the tuner's hardware AGC (what SDR++'s "Tuner AGC" does).
    /// `Some(db)` sets a manual gain in dB.
    pub sdr_gain_db: Option<f64>,
    pub sdr_lo_offset_hz: i64,
    pub audio_sample_rate: u32,
    pub fm_deviation_hz: u32,
    pub deemphasis_us: u32,
    pub web_bind_address: String,
    pub log_level: tracing::Level,
    pub channel_volume: f32,
    pub channels: Vec<Channel>,
}

impl AppConfig {
    pub fn load() -> Result<Self> {
        load_env_files()?;

        let channels = parse_channels(&std::env::var("CHANNELS").context("CHANNELS must be set")?)?;
        let sdr_sample_rate = std::env::var("SDR_SAMPLE_RATE")
            .context("SDR_SAMPLE_RATE must be set")?
            .parse()
            .context("SDR_SAMPLE_RATE must be a valid positive integer")?;
        let sdr_lo_offset_hz = std::env::var("SDR_LO_OFFSET_HZ")
            .context("SDR_LO_OFFSET_HZ must be set")?
            .parse()
            .context("SDR_LO_OFFSET_HZ must be a valid integer")?;
        let sdr_center_freq = derive_center_freq(&channels, sdr_lo_offset_hz)?;

        let config = AppConfig {
            sdr_backend: std::env::var("SDR_BACKEND").context("SDR_BACKEND must be set")?,
            sdr_device_args: std::env::var("SDR_DEVICE_ARGS").unwrap_or_default(),
            sdr_sample_rate,
            sdr_bandwidth_hz: match std::env::var("SDR_BANDWIDTH_HZ") {
                Ok(value) if !value.trim().is_empty() => Some(
                    value
                        .parse()
                        .context("SDR_BANDWIDTH_HZ must be a valid positive integer")?,
                ),
                _ => None,
            },
            sdr_center_freq,
            sdr_gain_db: match std::env::var("SDR_GAIN_DB")
                .context("SDR_GAIN_DB must be set")?
                .trim()
                .to_lowercase()
                .as_str()
            {
                "auto" | "agc" => None,
                value => Some(
                    value
                        .parse()
                        .context("SDR_GAIN_DB must be a number or \"auto\"")?,
                ),
            },
            sdr_lo_offset_hz,
            audio_sample_rate: std::env::var("AUDIO_SAMPLE_RATE")
                .context("AUDIO_SAMPLE_RATE must be set")?
                .parse()
                .context("AUDIO_SAMPLE_RATE must be a valid positive integer")?,
            fm_deviation_hz: std::env::var("FM_DEVIATION_HZ")
                .context("FM_DEVIATION_HZ must be set")?
                .parse()
                .context("FM_DEVIATION_HZ must be a valid positive integer")?,
            deemphasis_us: std::env::var("DEEMPHASIS_US")
                .context("DEEMPHASIS_US must be set")?
                .parse()
                .context("DEEMPHASIS_US must be a valid positive integer (e.g. 50 or 75)")?,
            web_bind_address: std::env::var("WEB_BIND_ADDRESS")
                .context("WEB_BIND_ADDRESS must be set")?,
            log_level: std::env::var("LOG_LEVEL")
                .unwrap_or_else(|_| "INFO".to_string())
                .parse::<tracing::Level>()
                .context(
                    "LOG_LEVEL must be a valid tracing level (TRACE, DEBUG, INFO, WARN, ERROR)",
                )?,
            channel_volume: std::env::var("CHANNEL_VOLUME")
                .context("CHANNEL_VOLUME must be set")?
                .parse()
                .context("CHANNEL_VOLUME must be a valid number")?,
            channels,
        };

        config.validate()?;
        Ok(config)
    }

    pub fn get() -> &'static AppConfig {
        static CONFIG: OnceLock<AppConfig> = OnceLock::new();
        CONFIG.get_or_init(|| AppConfig::load().expect("Failed to load application configuration"))
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(!self.channels.is_empty(), "CHANNELS has no entries");
        anyhow::ensure!(
            matches!(self.sdr_backend.as_str(), "soapy" | "synthetic"),
            "SDR_BACKEND must be \"soapy\" or \"synthetic\", got {:?}",
            self.sdr_backend
        );
        anyhow::ensure!(
            self.audio_sample_rate == 48_000,
            "AUDIO_SAMPLE_RATE must currently be 48000 (the DDC + resampler land on 48 kHz)"
        );
        anyhow::ensure!(self.fm_deviation_hz > 0, "FM_DEVIATION_HZ must be positive");

        // Each channel occupies the DDC low-pass cutoff on either side of its
        // centre, so the whole occupied band (offset +/- the channel
        // half-bandwidth) must fit inside the captured spectrum, not just the
        // centre frequency.
        let half_bandwidth = self.sdr_sample_rate as i64 / 2;
        let channel_half_bandwidth = CHANNEL_HALF_BANDWIDTH_HZ;
        let centre = self.sdr_center_freq as i64;
        for channel in &self.channels {
            let offset = channel.freq as i64 - centre;
            anyhow::ensure!(
                offset.abs() + channel_half_bandwidth < half_bandwidth,
                "channel {:?} at {} Hz is too close to or outside the captured spectrum edge (effective centre {} Hz +/- {} Hz usable after the {} Hz channel half-bandwidth, after SDR_LO_OFFSET_HZ = {}). Increase SDR_SAMPLE_RATE or reduce SDR_LO_OFFSET_HZ.",
                channel.name,
                channel.freq,
                centre,
                half_bandwidth,
                channel_half_bandwidth,
                self.sdr_lo_offset_hz
            );
        }
        Ok(())
    }
}

/// Load `.env.{ENVIRONMENT}` then override with `.env.{ENVIRONMENT}.local`.
/// The `cargo test` build uses the test files.
fn load_env_files() -> Result<()> {
    if cfg!(test) {
        let _ = dotenvy::from_filename(".env.test");
        let _ = dotenvy::from_filename_override(".env.test.local");
        return Ok(());
    }

    let environment = std::env::var("ENVIRONMENT").context("ENVIRONMENT must be set")?;
    match environment.as_str() {
        "development" | "test" | "production" => {
            let env_file = format!(".env.{environment}");
            dotenvy::from_filename(&env_file)
                .with_context(|| format!("Failed to load {env_file}"))?;
            let _ = dotenvy::from_filename_override(format!(".env.{environment}.local"));
            Ok(())
        }
        other => Err(anyhow::anyhow!("Unknown ENVIRONMENT value {other:?}")),
    }
}

/// Parse `CHANNELS` of the form `Name A|96500000, Name B|97300000`.
fn parse_channels(raw: &str) -> Result<Vec<Channel>> {
    raw.split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(|entry| {
            let (name, frequency) = entry.split_once('|').with_context(|| {
                format!("channel entry {entry:?} must be \"name|frequency_hz\"")
            })?;
            Ok(Channel {
                name: name.trim().to_string(),
                freq: frequency.trim().parse().with_context(|| {
                    format!(
                        "channel {name:?} frequency {frequency:?} must be a positive integer Hz"
                    )
                })?,
            })
        })
        .collect()
}

/// Midpoint of the channel frequencies, shifted by `lo_offset_hz` so the local
/// oscillator DC spike lands clear of every channel.
fn derive_center_freq(channels: &[Channel], lo_offset_hz: i64) -> Result<u64> {
    let minimum = channels
        .iter()
        .map(|channel| channel.freq)
        .min()
        .context("CHANNELS has no entries")?;
    let maximum = channels
        .iter()
        .map(|channel| channel.freq)
        .max()
        .expect("non-empty channels already checked above");
    let midpoint = minimum + (maximum - minimum) / 2;
    Ok((midpoint as i64).saturating_add(lo_offset_hz).max(0) as u64)
}
