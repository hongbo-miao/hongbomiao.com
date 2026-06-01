use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio_util::sync::CancellationToken;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;

mod channel;
mod config;
mod sdr;

mod digital_signal_processing {
    pub mod agc;
    pub mod audio_lowpass;
    pub mod ddc;
    pub mod de_emphasis;
    pub mod fm_demodulator;
    pub mod rational_resampler;
}

mod web {
    pub mod app_state;
    pub mod list_channels;
    pub mod serve_channel_audio;
    pub mod start_web_server;
}

use crate::config::AppConfig;
use crate::sdr::SyntheticTone;

#[tokio::main]
async fn main() -> Result<()> {
    let config = AppConfig::get();
    init_logging(config.log_level);
    info!("Config loaded: {config:?}");

    let cancel = CancellationToken::new();

    let synthetic_tones = build_synthetic_tones(config);
    let (sdr_source, sdr_task, hardware_thread) =
        sdr::start(config, synthetic_tones, cancel.clone())
            .await
            .context("starting SDR")?;

    let channels =
        channel::spawn_all(config, &sdr_source, cancel.clone()).context("spawning channels")?;
    info!("Running {} channels", channels.len());

    let endpoints = channels
        .iter()
        .map(|handle| handle.endpoint.clone())
        .collect();

    let static_dir = PathBuf::from("web");
    let web_task = web::start_web_server::start_web_server(
        &config.web_bind_address,
        static_dir,
        endpoints,
        cancel.clone(),
    )
    .await?;

    wait_for_shutdown_signal().await;
    info!("shutting down - closing SDR cleanly");
    cancel.cancel();

    let drain = async {
        for handle in channels {
            if let Err(error) = handle.task.await {
                error!("Channel task failed to join: {error}");
            }
        }
        if let Err(error) = sdr_task.await {
            error!("SDR ingest task failed to join: {error}");
        }
        if let Err(error) = web_task.await {
            error!("Web server task failed to join: {error}");
        }
        // The SoapySDR device lives on its own OS thread. Join it so the device
        // is dropped (and the radio closed) before the process exits; otherwise
        // the hardware can be left wedged until it is physically replugged.
        if let Some(hardware_thread) = hardware_thread {
            match tokio::task::spawn_blocking(move || hardware_thread.join()).await {
                Ok(Ok(())) => {}
                Ok(Err(panic)) => error!("SDR hardware thread panicked: {panic:?}"),
                Err(error) => error!("Joining SDR hardware thread failed: {error}"),
            }
        }
    };
    let _ = tokio::time::timeout(Duration::from_secs(5), drain).await;

    Ok(())
}

/// Wait for either Ctrl-C (SIGINT) or SIGTERM. Handling SIGTERM matters because
/// the `just` recipes send it on Ctrl-C; without it the process would die before
/// the SDR device closes cleanly, leaving the Airspy wedged until replug.
async fn wait_for_shutdown_signal() {
    use tokio::signal::unix::{SignalKind, signal};
    let mut interrupt = signal(SignalKind::interrupt()).expect("installing SIGINT handler");
    let mut terminate = signal(SignalKind::terminate()).expect("installing SIGTERM handler");
    tokio::select! {
        _ = interrupt.recv() => {}
        _ = terminate.recv() => {}
    }
}

/// One FM test tone per channel, placed at the channel's offset from the SDR
/// centre. Only used by the `synthetic` backend; real backends ignore it.
fn build_synthetic_tones(config: &AppConfig) -> Vec<SyntheticTone> {
    config
        .channels
        .iter()
        .enumerate()
        .map(|(index, channel)| SyntheticTone {
            offset_hz: channel.freq as f64 - config.sdr_center_freq as f64,
            tone_hz: 400.0 + 200.0 * index as f32,
            deviation_hz: config.fm_deviation_hz as f32,
        })
        .collect()
}

fn init_logging(level: tracing::Level) {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("radio_streamer={level},warn")));
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .with(filter)
        .init();
}
