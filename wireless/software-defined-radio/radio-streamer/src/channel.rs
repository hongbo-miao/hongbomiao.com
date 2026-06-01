//! One actor per configured channel. Subscribes to the shared wideband I/Q
//! stream, runs the FM DSP chain (DDC -> FM demod -> de-emphasis ->
//! audio low-pass -> rational resample -> AGC -> volume), and broadcasts the
//! resulting 48 kHz mono audio so any number of browser listeners can subscribe.

use std::sync::Arc;

use anyhow::{Context, Result};
use num_complex::Complex;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::config::{AppConfig, Channel};
use crate::digital_signal_processing::agc::Agc;
use crate::digital_signal_processing::audio_lowpass::AudioLowpass;
use crate::digital_signal_processing::ddc::Ddc;
use crate::digital_signal_processing::de_emphasis::DeEmphasis;
use crate::digital_signal_processing::fm_demodulator::FmDemodulator;
use crate::digital_signal_processing::rational_resampler::RationalResampler;
use crate::sdr::IqChunk;

/// One block of post-DSP mono audio at the configured audio sample rate.
pub type AudioFrame = Arc<Vec<f32>>;

/// DDC low-pass cut-off, applied on the complex baseband — so the resulting
/// channel bandwidth is `2 * DDC_LOWPASS_HZ`. 100 kHz here = 200 kHz channel
/// BW, matching SDR++'s WBFM "Bandwidth 200000" and the broadcast-FM Carson
/// bandwidth (~180 kHz mono). Must stay below the post-decimation Nyquist.
const DDC_LOWPASS_HZ: f32 = 100_000.0;

/// Audio low-pass cut-off (mono FM tops out at 15 kHz; above that sits the
/// 19 kHz stereo pilot and SCA subcarriers).
const AUDIO_LOWPASS_HZ: f32 = 15_000.0;

/// AGC target envelope. 0.25 leaves headroom below 0 dBFS for the i16 conversion.
const AGC_TARGET: f32 = 0.25;

/// AGC ceiling. FM audio level varies with modulation depth across stations;
/// a moderate boost evens them out.
const AGC_MAX_GAIN: f32 = 8.0;

/// Audio broadcast depth (frames). A slow browser client drops frames rather
/// than back-pressuring the DSP loop.
const AUDIO_BROADCAST_DEPTH: usize = 64;

/// Everything the web layer needs to expose a channel: identity plus a handle
/// to subscribe to its audio.
#[derive(Clone)]
pub struct ChannelEndpoint {
    pub id: usize,
    pub name: String,
    pub freq: u64,
    pub sample_rate: u32,
    pub audio_sender: broadcast::Sender<AudioFrame>,
}

pub struct ChannelHandle {
    pub endpoint: ChannelEndpoint,
    pub task: tokio::task::JoinHandle<()>,
}

#[allow(clippy::too_many_arguments)]
fn spawn_channel(
    id: usize,
    channel: &Channel,
    config: &AppConfig,
    sdr_sample_rate: u32,
    sdr_center_freq: u64,
    audio_sample_rate: u32,
    iq_receiver: broadcast::Receiver<IqChunk>,
    cancel: CancellationToken,
) -> Result<ChannelHandle> {
    let name = channel.name.clone();
    let freq = channel.freq;

    // The DDC decimates by an integer to an intermediate rate that divides the
    // SDR rate; the rational resampler then converts that to the exact audio
    // rate (e.g. Airspy R2 10_000_000 -> 400_000 -> 48_000).
    let (decimation, intermediate_rate) =
        choose_intermediate_rate(sdr_sample_rate, audio_sample_rate)
            .with_context(|| {
                format!(
                    "no usable intermediate rate for SDR sample_rate {sdr_sample_rate} -> audio {audio_sample_rate}"
                )
            })?;
    let divisor = gcd(audio_sample_rate, intermediate_rate);
    let interpolation = (audio_sample_rate / divisor) as usize;
    let decimation_ratio = (intermediate_rate / divisor) as usize;

    let offset_hz = freq as f64 - sdr_center_freq as f64;
    let downconverter = Ddc::new(
        sdr_sample_rate,
        intermediate_rate,
        offset_hz,
        DDC_LOWPASS_HZ,
    );
    let demodulator = FmDemodulator::new(intermediate_rate, config.fm_deviation_hz);
    let de_emphasis = DeEmphasis::new(intermediate_rate, config.deemphasis_us);
    let audio_lowpass = AudioLowpass::new(intermediate_rate, AUDIO_LOWPASS_HZ);
    let resampler = RationalResampler::new(interpolation, decimation_ratio);
    let agc = Agc::new(audio_sample_rate, AGC_TARGET, AGC_MAX_GAIN);
    let volume = config.channel_volume.clamp(0.0, 2.0);

    let (audio_sender, _) = broadcast::channel::<AudioFrame>(AUDIO_BROADCAST_DEPTH);

    let endpoint = ChannelEndpoint {
        id,
        name: name.clone(),
        freq,
        sample_rate: audio_sample_rate,
        audio_sender: audio_sender.clone(),
    };

    let task = tokio::spawn(run_channel(
        name.clone(),
        volume,
        downconverter,
        demodulator,
        de_emphasis,
        audio_lowpass,
        resampler,
        agc,
        iq_receiver,
        audio_sender,
        cancel,
    ));

    info!(
        "Channel {name} started at {freq} Hz (offset {offset_hz} Hz, decimation {decimation} -> {intermediate_rate} Hz, resample {interpolation}/{decimation_ratio})"
    );

    Ok(ChannelHandle { endpoint, task })
}

/// Greatest common divisor.
fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Pick an integer DDC decimation whose resulting intermediate rate divides the
/// SDR rate, lands in a comfortable window for FM (wide enough to fit ~200 kHz
/// of channel bandwidth), and gives the smallest rational resampling
/// denominator into `audio_rate`. Returns (decimation, intermediate_rate).
fn choose_intermediate_rate(sdr_sample_rate: u32, audio_rate: u32) -> Option<(usize, u32)> {
    // Intermediate rate must satisfy Nyquist > DDC_LOWPASS_HZ (100 kHz).
    // Airspy R2 at 10 MS/s lands cleanly on 400 kHz via decim 25.
    let minimum_rate = 240_000_u32;
    let maximum_rate = 500_000_u32;

    let lowest_decimation = (sdr_sample_rate / maximum_rate).max(1);
    let highest_decimation = (sdr_sample_rate / minimum_rate).max(1);

    let mut best: Option<(usize, u32, u32)> = None; // (decimation, intermediate, denominator)
    for decimation in lowest_decimation..=highest_decimation {
        if !sdr_sample_rate.is_multiple_of(decimation) {
            continue;
        }
        let intermediate = sdr_sample_rate / decimation;
        if intermediate < minimum_rate || intermediate > maximum_rate {
            continue;
        }
        let denominator = intermediate / gcd(audio_rate, intermediate);
        let is_better = match best {
            None => true,
            Some((best_decimation, _, best_denominator)) => {
                denominator < best_denominator
                    || (denominator == best_denominator && decimation as usize > best_decimation)
            }
        };
        if is_better {
            best = Some((decimation as usize, intermediate, denominator));
        }
    }

    best.map(|(decimation, intermediate, _)| (decimation, intermediate))
}

#[allow(clippy::too_many_arguments)]
async fn run_channel(
    channel_name: String,
    volume: f32,
    mut downconverter: Ddc,
    mut demodulator: FmDemodulator,
    mut de_emphasis: DeEmphasis,
    mut audio_lowpass: AudioLowpass,
    mut resampler: RationalResampler,
    mut agc: Agc,
    mut iq_receiver: broadcast::Receiver<IqChunk>,
    audio_sender: broadcast::Sender<AudioFrame>,
    cancel: CancellationToken,
) {
    let mut baseband: Vec<Complex<f32>> = Vec::with_capacity(4096);
    let mut intermediate_audio: Vec<f32> = Vec::with_capacity(4096);
    let mut audio: Vec<f32> = Vec::with_capacity(4096);
    let mut last_diagnostic = std::time::Instant::now();

    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                debug!("Channel {channel_name} task exiting");
                break;
            }
            received = iq_receiver.recv() => {
                let chunk: IqChunk = match received {
                    Ok(chunk) => chunk,
                    Err(broadcast::error::RecvError::Lagged(count)) => {
                        warn!("Channel {channel_name} lagged {count} chunks behind SDR ingest");
                        continue;
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        debug!("SDR broadcast closed for channel {channel_name}");
                        break;
                    }
                };

                // Stage 1: digital downconvert to the intermediate baseband I/Q.
                baseband.clear();
                downconverter.process(&chunk, &mut baseband);

                // Stage 2: FM polar-discriminator demodulation.
                intermediate_audio.clear();
                demodulator.process(&baseband, &mut intermediate_audio);

                // Stage 3: de-emphasis (inverse of the broadcaster's high-frequency boost).
                de_emphasis.process_in_place(&mut intermediate_audio);

                // Stage 4: audio low-pass at 15 kHz, run at the intermediate
                // rate so the cutoff is well below Nyquist (clean rejection of
                // the 19 kHz stereo pilot before resampling).
                audio_lowpass.process_in_place(&mut intermediate_audio);

                // Stage 5: rational resample to the exact audio rate (48 kHz).
                audio.clear();
                resampler.process(&intermediate_audio, &mut audio);

                // Stage 6: AGC. FM audio level varies with station modulation
                // depth; the AGC normalises across channels.
                agc.process_in_place(&mut audio);

                // Stage 7: per-channel volume.
                if volume != 1.0 {
                    for sample in audio.iter_mut() {
                        *sample *= volume;
                    }
                }

                if last_diagnostic.elapsed() >= std::time::Duration::from_secs(5) {
                    let channel_baseband_level = rms_complex(&baseband);
                    let final_audio_level = rms_real(&audio);
                    info!(
                        "{channel_name}  channel_signal={channel_baseband_level:.3}  audio_out={final_audio_level:.3}"
                    );
                    last_diagnostic = std::time::Instant::now();
                }

                // Broadcast to listeners. Err just means no browser is
                // connected. Move the buffer into the Arc instead of cloning it;
                // a fresh Vec takes its place for the next iteration, so the hot
                // path costs one allocation rather than allocation plus a copy.
                let mut frame = Vec::with_capacity(audio.capacity());
                std::mem::swap(&mut frame, &mut audio);
                let _ = audio_sender.send(Arc::new(frame));
            }
        }
    }
}

fn rms_complex(samples: &[Complex<f32>]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|sample| sample.norm_sqr()).sum::<f32>() / samples.len() as f32).sqrt()
}

fn rms_real(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32).sqrt()
}

/// Build every channel described in config.
pub fn spawn_all(
    config: &AppConfig,
    sdr: &crate::sdr::SdrSource,
    cancel: CancellationToken,
) -> Result<Vec<ChannelHandle>> {
    config
        .channels
        .iter()
        .enumerate()
        .map(|(id, channel)| {
            spawn_channel(
                id,
                channel,
                config,
                sdr.sample_rate,
                sdr.center_freq,
                config.audio_sample_rate,
                sdr.subscribe(),
                cancel.clone(),
            )
        })
        .collect()
}
