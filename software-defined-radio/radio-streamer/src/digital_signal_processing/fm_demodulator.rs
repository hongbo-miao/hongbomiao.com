//! FM polar discriminator. For an FM signal at baseband,
//!   z[n] = amplitude * exp(j * phase[n])
//! and the instantaneous frequency is the derivative of `phase`.
//! Per-sample phase delta:
//!   arg(z[n] * conj(z[n-1])) = phase[n] - phase[n-1]
//! Dividing by the per-sample phase step at peak deviation normalises the
//! output to ~±1.0 at peak modulation. FM is amplitude-independent by design:
//! |z| can vary freely (limiter at TX) but the phase carries the audio.

use num_complex::Complex;

pub struct FmDemodulator {
    previous_sample: Complex<f32>,
    output_scale: f32,
}

impl FmDemodulator {
    pub fn new(sample_rate: u32, peak_deviation_hz: u32) -> Self {
        let max_phase_step = std::f32::consts::TAU * peak_deviation_hz as f32 / sample_rate as f32;
        Self {
            previous_sample: Complex::new(0.0, 0.0),
            output_scale: 1.0 / max_phase_step.max(1e-9),
        }
    }

    pub fn process(&mut self, iq_samples: &[Complex<f32>], audio_out: &mut Vec<f32>) {
        audio_out.reserve(iq_samples.len());
        for sample in iq_samples {
            let product = *sample * self.previous_sample.conj();
            let phase_delta = product.im.atan2(product.re);
            audio_out.push(phase_delta * self.output_scale);
            self.previous_sample = *sample;
        }
    }
}

#[cfg(test)]
#[path = "fm_demodulator_test.rs"]
mod tests;
