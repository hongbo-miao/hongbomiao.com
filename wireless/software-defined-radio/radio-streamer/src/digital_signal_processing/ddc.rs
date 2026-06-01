//! Digital downconverter: mix a channel down to DC, low-pass filter, decimate.
//! Polyphase trick: pre-rotate the real LPF taps by the NCO frequency so
//! `mix + FIR + decimate` collapses into one complex MAC loop, with no
//! input-rate NCO multiply.
//!   complex_taps[k] = lowpass_taps[k] * exp(j * nco_phase_step * k)
//!   y[m] = exp(j * nco_phase_step * m * decimation) *
//!          sum_k complex_taps[k] * iq[m*decimation + k]

use std::f32::consts::TAU;

use num_complex::Complex;

const FIR_LEN: usize = 257;

pub struct Ddc {
    complex_taps: Vec<Complex<f32>>,
    output_phasor: Complex<f32>,
    output_phasor_step: Complex<f32>,
    decimation: usize,
    history: Vec<Complex<f32>>,
    decimation_phase: usize,
    // Reused across calls (history + current chunk) so the hot path does not
    // allocate a fresh Vec on every chunk.
    combined_buffer: Vec<Complex<f32>>,
}

impl Ddc {
    pub fn new(
        input_sample_rate: u32,
        output_sample_rate: u32,
        channel_offset_hz: f64,
        lowpass_cutoff_hz: f32,
    ) -> Self {
        assert!(input_sample_rate.is_multiple_of(output_sample_rate));
        assert!(lowpass_cutoff_hz < (output_sample_rate as f32) / 2.0);
        let decimation = (input_sample_rate / output_sample_rate) as usize;

        let nco_phase_step = (-TAU * channel_offset_hz as f32) / (input_sample_rate as f32);

        let lowpass_taps = design_lowpass_fir(FIR_LEN, lowpass_cutoff_hz, input_sample_rate as f32);
        let complex_taps: Vec<Complex<f32>> = lowpass_taps
            .iter()
            .enumerate()
            .map(|(tap_index, &tap)| {
                let theta = nco_phase_step * tap_index as f32;
                let (sin_theta, cos_theta) = theta.sin_cos();
                Complex::new(tap * cos_theta, tap * sin_theta)
            })
            .collect();

        let initial_phase = -nco_phase_step * (FIR_LEN - 1) as f32;
        let (sin_initial, cos_initial) = initial_phase.sin_cos();
        let output_phasor = Complex::new(cos_initial, sin_initial);

        let step_phase = nco_phase_step * decimation as f32;
        let (sin_step, cos_step) = step_phase.sin_cos();
        let output_phasor_step = Complex::new(cos_step, sin_step);

        Self {
            complex_taps,
            output_phasor,
            output_phasor_step,
            decimation,
            history: vec![Complex::new(0.0, 0.0); FIR_LEN - 1],
            decimation_phase: 0,
            combined_buffer: Vec::with_capacity(FIR_LEN + crate::sdr::CHUNK_SIZE),
        }
    }

    pub fn process(&mut self, iq_samples: &[Complex<f32>], baseband_out: &mut Vec<Complex<f32>>) {
        let fir_len = self.complex_taps.len();

        self.combined_buffer.clear();
        self.combined_buffer.extend_from_slice(&self.history);
        self.combined_buffer.extend_from_slice(iq_samples);

        let valid_end = self.combined_buffer.len().saturating_sub(fir_len - 1);
        let mut window_start = self.decimation_phase;
        while window_start < valid_end {
            let window = &self.combined_buffer[window_start..window_start + fir_len];
            let mut accumulator = Complex::new(0.0_f32, 0.0_f32);
            for (complex_tap, sample) in self.complex_taps.iter().zip(window.iter()) {
                accumulator += complex_tap * sample;
            }
            baseband_out.push(accumulator * self.output_phasor);

            self.output_phasor *= self.output_phasor_step;
            window_start += self.decimation;
        }

        // Renormalise: complex multiplies accumulate f32 magnitude drift.
        let magnitude_squared = self.output_phasor.norm_sqr();
        if (magnitude_squared - 1.0).abs() > 1e-6 {
            self.output_phasor /= magnitude_squared.sqrt();
        }

        self.decimation_phase = window_start - valid_end;

        let tail_start = self.combined_buffer.len() - (fir_len - 1);
        self.history
            .copy_from_slice(&self.combined_buffer[tail_start..]);
    }
}

/// Windowed-sinc low-pass with a Hann window. Odd length -> linear phase
/// (integer group delay). Normalised so the taps sum to 1 (unity DC gain).
fn design_lowpass_fir(tap_count: usize, cutoff_hz: f32, sample_rate: f32) -> Vec<f32> {
    assert!(tap_count % 2 == 1);
    let last_index = (tap_count - 1) as f32;
    let normalised_cutoff = cutoff_hz / sample_rate;
    let mut taps = Vec::with_capacity(tap_count);

    for index in 0..tap_count {
        let centred = index as f32 - last_index / 2.0;
        let sinc = if centred == 0.0 {
            2.0 * normalised_cutoff
        } else {
            (TAU * normalised_cutoff * centred).sin() / (std::f32::consts::PI * centred)
        };
        let hann_window = 0.5 * (1.0 - (TAU * index as f32 / last_index).cos());
        taps.push(sinc * hann_window);
    }

    let dc_gain: f32 = taps.iter().sum();
    for tap in &mut taps {
        *tap /= dc_gain;
    }
    taps
}

#[cfg(test)]
#[path = "ddc_test.rs"]
mod tests;
