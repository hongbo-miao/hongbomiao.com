//! Wraps the SDR backends behind one async stream of I/Q chunks and fans that
//! stream out to every channel via a broadcast channel.
//!
//! Backends: `soapy` (SoapySDR — Airspy R2 and friends) and `synthetic`
//! (FM-modulated test tones, one per channel, so the whole capture -> DSP ->
//! browser path can be exercised with no hardware).

use std::pin::Pin;
use std::sync::Arc;

use anyhow::{Context, Result};
use futures::{Stream, StreamExt};
use num_complex::Complex;
use tokio::sync::{broadcast, mpsc};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::config::AppConfig;

/// Unified I/Q stream type. Every backend reduces to this.
type IqStream = Pin<Box<dyn Stream<Item = anyhow::Result<Vec<Complex<f32>>>> + Send>>;

/// One chunk of complex samples, shared across many subscribers cheaply.
pub type IqChunk = Arc<Vec<Complex<f32>>>;

/// Number of complex samples pulled per chunk. 16384 @ 2.4 MS/s is about 6.8 ms.
pub const CHUNK_SIZE: usize = 16_384;

/// Broadcast depth. Subscribers that fall further behind than this see Lagged
/// and skip, rather than back-pressuring the SDR ingest task.
const BROADCAST_DEPTH: usize = 32;

/// A synthetic FM test signal at a given offset from the capture centre.
#[derive(Debug, Clone, Copy)]
pub struct SyntheticTone {
    /// Carrier offset from the SDR centre frequency, in Hz.
    pub offset_hz: f64,
    /// Audio modulation frequency, in Hz (what you hear once demodulated).
    pub tone_hz: f32,
    /// Peak FM deviation, in Hz. Should match `FM_DEVIATION_HZ` so the demod's
    /// scaling matches the synthetic signal.
    pub deviation_hz: f32,
}

pub struct SdrSource {
    pub sample_rate: u32,
    pub center_freq: u64,
    pub iq_sender: broadcast::Sender<IqChunk>,
}

impl SdrSource {
    pub fn subscribe(&self) -> broadcast::Receiver<IqChunk> {
        self.iq_sender.subscribe()
    }
}

/// Spawn the SDR ingest task. Returns the source handle (so channels can
/// subscribe), a tokio JoinHandle for the ingest task, and - for hardware
/// backends - the OS thread that owns the device, so shutdown can join it and
/// guarantee the radio is closed before the process exits.
pub async fn start(
    config: &AppConfig,
    synthetic_tones: Vec<SyntheticTone>,
    cancel: CancellationToken,
) -> Result<(
    SdrSource,
    tokio::task::JoinHandle<()>,
    Option<std::thread::JoinHandle<()>>,
)> {
    let (iq_sender, _) = broadcast::channel::<IqChunk>(BROADCAST_DEPTH);

    let (reader, hardware_thread) = open_reader(config, &synthetic_tones).await?;

    info!(
        "SDR opened: backend {}, centre {} Hz, sample rate {} Hz, lo_offset {} Hz, gain {}",
        config.sdr_backend,
        config.sdr_center_freq,
        config.sdr_sample_rate,
        config.sdr_lo_offset_hz,
        config
            .sdr_gain_db
            .map(|gain| format!("{gain} dB"))
            .unwrap_or_else(|| "auto (tuner AGC)".to_string()),
    );

    let sample_rate = config.sdr_sample_rate;
    let center_freq = config.sdr_center_freq;
    let sender_for_task = iq_sender.clone();
    let handle = tokio::spawn(async move {
        run_ingest(reader, sender_for_task, cancel).await;
    });

    Ok((
        SdrSource {
            sample_rate,
            center_freq,
            iq_sender,
        },
        handle,
        hardware_thread,
    ))
}

/// Open the configured backend. Returns the I/Q stream plus, for hardware
/// backends that run the device on a dedicated OS thread, that thread's handle.
async fn open_reader(
    config: &AppConfig,
    synthetic_tones: &[SyntheticTone],
) -> Result<(IqStream, Option<std::thread::JoinHandle<()>>)> {
    match config.sdr_backend.as_str() {
        "soapy" => {
            let args = if config.sdr_device_args.is_empty() {
                "driver=airspy".to_string()
            } else {
                config.sdr_device_args.clone()
            };
            let (stream, hardware_thread) = open_soapy(args, config).await?;
            Ok((stream, Some(hardware_thread)))
        }
        "synthetic" => {
            info!(
                "Synthetic I/Q source ({} test tones) - no hardware required",
                synthetic_tones.len()
            );
            Ok((
                synthetic_stream(config.sdr_sample_rate, synthetic_tones.to_vec()),
                None,
            ))
        }
        other => anyhow::bail!("unknown SDR backend {:?}", other),
    }
}

/// Real-time paced stream summing one FM-modulated carrier per configured tone,
/// plus a low noise floor. Each channel's DDC isolates its carrier and the FM
/// demodulator recovers a distinct audio tone, so you can hear every channel.
fn synthetic_stream(sample_rate: u32, tones: Vec<SyntheticTone>) -> IqStream {
    let chunk_nanos = (CHUNK_SIZE as u64 * 1_000_000_000) / sample_rate as u64;
    let mut interval = tokio::time::interval(std::time::Duration::from_nanos(chunk_nanos));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    let sample_period = 1.0_f64 / sample_rate as f64;
    // Phase accumulators kept across chunks so each carrier stays continuous.
    let phases: Vec<f64> = vec![0.0; tones.len()];
    let start_index: u64 = 0;

    Box::pin(futures::stream::unfold(
        (interval, start_index, phases),
        move |(mut interval, mut index, mut phases)| {
            let tones = tones.clone();
            async move {
                interval.tick().await;
                let mut samples = Vec::with_capacity(CHUNK_SIZE);
                for _ in 0..CHUNK_SIZE {
                    let time = index as f64 * sample_period;
                    let mut sample = Complex::new(0.0_f32, 0.0_f32);
                    for (tone_index, tone) in tones.iter().enumerate() {
                        // FM: instantaneous frequency = offset_hz + deviation_hz * sin(2 pi f_tone t).
                        // Integrate (Euler) to phase.
                        let instantaneous_hz = tone.offset_hz
                            + tone.deviation_hz as f64
                                * (std::f64::consts::TAU * tone.tone_hz as f64 * time).sin();
                        // Wrap to [0, TAU) so the accumulator does not lose
                        // float precision over long continuous runs.
                        phases[tone_index] = (phases[tone_index]
                            + std::f64::consts::TAU * instantaneous_hz * sample_period)
                            .rem_euclid(std::f64::consts::TAU);
                        let amplitude = 0.3_f32;
                        sample += Complex::new(
                            amplitude * phases[tone_index].cos() as f32,
                            amplitude * phases[tone_index].sin() as f32,
                        );
                    }
                    samples.push(sample);
                    index = index.wrapping_add(1);
                }
                Some((Ok(samples), (interval, index, phases)))
            }
        },
    ))
}

/// Pull the `driver=...` token out of a SoapySDR device-args string.
fn parse_soapy_driver(args: &str) -> &str {
    if args.is_empty() {
        return "airspy";
    }
    for token in args.split(',') {
        if let Some(value) = token.trim().strip_prefix("driver=") {
            return value;
        }
    }
    ""
}

/// Open a SoapySDR device, configure it for wideband capture, and return it as
/// an async I/Q stream plus the OS thread that owns the device (so shutdown can
/// join it and guarantee the radio is closed before the process exits).
async fn open_soapy(
    args: String,
    config: &AppConfig,
) -> Result<(IqStream, std::thread::JoinHandle<()>)> {
    use soapysdr::Direction::Rx;
    let center_freq = config.sdr_center_freq as f64;
    let sample_rate = config.sdr_sample_rate;
    let gain = config.sdr_gain_db;
    let driver = parse_soapy_driver(&args).to_string();
    let bandwidth_hz = config.sdr_bandwidth_hz;

    let setup_args = args.clone();
    let setup_driver = driver.clone();
    // Open + configure on a blocking thread — SoapySDR's `Device::new` does
    // synchronous USB I/O that can take hundreds of milliseconds.
    let device = tokio::task::spawn_blocking(move || -> Result<soapysdr::Device> {
        let device = soapysdr::Device::new(setup_args.as_str())
            .context("opening SoapySDR device - check `SoapySDRUtil --find`")?;
        device
            .set_sample_rate(Rx, 0, sample_rate as f64)
            .context("setting SoapySDR sample rate")?;
        device
            .set_frequency(Rx, 0, center_freq, ())
            .context("setting SoapySDR centre frequency")?;
        if let Some(bandwidth) = bandwidth_hz {
            match device.set_bandwidth(Rx, 0, bandwidth as f64) {
                Ok(()) => {
                    let actual = device.bandwidth(Rx, 0).unwrap_or(bandwidth as f64);
                    info!(
                        "Analog filter bandwidth set for driver {setup_driver}: requested {bandwidth} Hz, actual {actual} Hz"
                    );
                }
                Err(error) => warn!(
                    "set_bandwidth failed for driver {setup_driver} at {bandwidth} Hz, continuing with driver default: {error}"
                ),
            }
        }
        match gain {
            Some(gain_db) => {
                device
                    .set_gain_mode(Rx, 0, false)
                    .context("disabling SoapySDR AGC")?;
                device
                    .set_gain(Rx, 0, gain_db)
                    .context("setting SoapySDR gain")?;
            }
            None => {
                device
                    .set_gain_mode(Rx, 0, true)
                    .context("enabling SoapySDR AGC")?;
            }
        }
        Ok(device)
    })
    .await
    .context("SoapySDR setup task panicked")??;

    // The rx_stream is `!Send`, so it lives on its own OS thread and samples
    // flow out through a tokio mpsc.
    let (sender, receiver) = mpsc::channel::<anyhow::Result<Vec<Complex<f32>>>>(32);
    let driver_for_thread = driver.clone();
    let hardware_thread = std::thread::spawn(move || {
        if let Err(error) = run_soapy_stream(device, sender.clone()) {
            error!("SoapySDR rx loop for driver {driver_for_thread} exited with error: {error}");
            let _ = sender.blocking_send(Err(error));
        }
    });

    let stream: IqStream = Box::pin(futures::stream::unfold(
        receiver,
        |mut receiver| async move { receiver.recv().await.map(|item| (item, receiver)) },
    ));
    Ok((stream, hardware_thread))
}

/// Drain the SoapySDR RX stream into the tokio channel. Runs on its own OS thread.
fn run_soapy_stream(
    device: soapysdr::Device,
    sender: mpsc::Sender<anyhow::Result<Vec<Complex<f32>>>>,
) -> Result<()> {
    let mut stream = device
        .rx_stream::<Complex<i16>>(&[0])
        .context("creating SoapySDR rx stream")?;
    let max_transfer_unit = stream.mtu().unwrap_or(16_384);
    stream
        .activate(None)
        .context("activating SoapySDR rx stream")?;
    let mut buffer = vec![Complex::new(0i16, 0i16); max_transfer_unit];
    loop {
        // Short read timeout (200 ms) so that when the consumer goes away on
        // shutdown, this loop notices quickly and drops the stream/device.
        match stream.read(&mut [&mut buffer], 200_000) {
            Ok(0) => {
                let _ = sender.blocking_send(Err(anyhow::anyhow!("SoapySDR stream ended")));
                break;
            }
            Ok(length) => {
                let samples: Vec<Complex<f32>> = buffer[..length]
                    .iter()
                    .map(|value| {
                        Complex::new(value.re as f32 / 32_768.0, value.im as f32 / 32_768.0)
                    })
                    .collect();
                if sender.blocking_send(Ok(samples)).is_err() {
                    break;
                }
            }
            Err(error) if error.code == soapysdr::ErrorCode::Timeout => continue,
            Err(error) => {
                let _ =
                    sender.blocking_send(Err(anyhow::Error::new(error).context("SoapySDR read")));
                break;
            }
        }
    }
    let _ = device;
    Ok(())
}

async fn run_ingest(
    mut reader: IqStream,
    iq_sender: broadcast::Sender<IqChunk>,
    cancel: CancellationToken,
) {
    let mut dropped_chunk_count: u64 = 0;
    let mut report_interval = tokio::time::interval(std::time::Duration::from_secs(30));
    report_interval.tick().await; // skip the immediate first tick

    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                info!("SDR ingest cancelled");
                break;
            }
            _ = report_interval.tick() => {
                if dropped_chunk_count > 0 {
                    warn!("Dropped {dropped_chunk_count} SDR chunks (no subscribers or all lagged) in the last 30s");
                    dropped_chunk_count = 0;
                }
            }
            maybe = reader.next() => {
                match maybe {
                    Some(Ok(samples)) => {
                        if iq_sender.send(Arc::new(samples)).is_err() {
                            dropped_chunk_count += 1;
                        }
                    }
                    Some(Err(error)) => {
                        error!("SDR read error: {error:#}");
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    }
                    None => {
                        debug!("SDR stream ended");
                        break;
                    }
                }
            }
        }
    }
}
