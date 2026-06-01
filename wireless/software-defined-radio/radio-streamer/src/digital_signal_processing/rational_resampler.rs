//! Rational sample-rate conversion (interpolate by L, decimate by M) via a
//! polyphase FIR. Conceptually: insert L-1 zeros between input samples,
//! low-pass at min(Fs_in, Fs_out)/2, keep every M-th sample. The polyphase
//! reorganisation skips the zero multiplies entirely: tap bank `p` uses taps
//! `p, p+L, p+2L, ...`, and output `n` selects phase `(n*M) mod L` against
//! the input sample at index `floor(n*M / L)`.

use std::f32::consts::{PI, TAU};

pub struct RationalResampler {
    interpolation_factor: usize,
    decimation_factor: usize,
    taps: Vec<f32>,
    taps_per_phase: usize,
    // Circular buffer: `write_index` holds the newest sample, `write_index - k`
    // (wrapping) holds the sample `k` steps ago. Avoids an O(taps_per_phase)
    // shift per input sample.
    input_history: Vec<f32>,
    write_index: usize,
    input_index: i64,
    output_index: i64,
    is_identity: bool,
}

impl RationalResampler {
    pub fn new(interpolation_factor: usize, decimation_factor: usize) -> Self {
        assert!(interpolation_factor >= 1 && decimation_factor >= 1);

        if interpolation_factor == 1 && decimation_factor == 1 {
            return Self {
                interpolation_factor: 1,
                decimation_factor: 1,
                taps: Vec::new(),
                taps_per_phase: 0,
                input_history: Vec::new(),
                write_index: 0,
                input_index: 0,
                output_index: 0,
                is_identity: true,
            };
        }

        let max_factor = interpolation_factor.max(decimation_factor);
        let taps_per_phase = 32;
        let total_taps = taps_per_phase * interpolation_factor;
        let normalised_cutoff = 0.5 / max_factor as f32;
        let taps =
            design_polyphase_lowpass(total_taps, normalised_cutoff, interpolation_factor as f32);

        Self {
            interpolation_factor,
            decimation_factor,
            taps,
            taps_per_phase,
            input_history: vec![0.0; taps_per_phase],
            write_index: 0,
            input_index: 0,
            output_index: 0,
            is_identity: false,
        }
    }

    pub fn process(&mut self, audio_samples: &[f32], audio_out: &mut Vec<f32>) {
        if self.is_identity {
            audio_out.extend_from_slice(audio_samples);
            return;
        }

        let interpolation = self.interpolation_factor as i64;
        let decimation = self.decimation_factor as i64;

        for &sample in audio_samples {
            self.input_history[self.write_index] = sample;
            let current_input = self.input_index;

            loop {
                let upsampled_position = self.output_index * decimation;
                let aligned_input = upsampled_position / interpolation;
                if aligned_input != current_input {
                    break;
                }
                let phase = (upsampled_position % interpolation) as usize;
                let mut accumulator = 0.0_f32;
                for tap_within_phase in 0..self.taps_per_phase {
                    let tap_index = phase + tap_within_phase * self.interpolation_factor;
                    let history_index = if self.write_index >= tap_within_phase {
                        self.write_index - tap_within_phase
                    } else {
                        self.write_index + self.taps_per_phase - tap_within_phase
                    };
                    accumulator += self.taps[tap_index] * self.input_history[history_index];
                }
                audio_out.push(accumulator);
                self.output_index += 1;
            }

            self.write_index = (self.write_index + 1) % self.taps_per_phase;
            self.input_index += 1;
        }
    }
}

fn design_polyphase_lowpass(
    total_taps: usize,
    normalised_cutoff: f32,
    passband_gain: f32,
) -> Vec<f32> {
    let last_index = (total_taps - 1) as f32;
    let mut taps = Vec::with_capacity(total_taps);
    for index in 0..total_taps {
        let offset = index as f32 - last_index / 2.0;
        let sinc = if offset == 0.0 {
            2.0 * normalised_cutoff
        } else {
            (TAU * normalised_cutoff * offset).sin() / (PI * offset)
        };
        let hann_window = 0.5 * (1.0 - (TAU * index as f32 / last_index).cos());
        taps.push(sinc * hann_window);
    }
    let sum: f32 = taps.iter().sum();
    for tap in &mut taps {
        *tap = *tap / sum * passband_gain;
    }
    taps
}

#[cfg(test)]
#[path = "rational_resampler_test.rs"]
mod tests;
