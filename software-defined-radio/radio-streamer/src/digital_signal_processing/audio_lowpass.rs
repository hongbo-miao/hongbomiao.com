//! RBJ-cookbook second-order biquad low-pass, Direct Form II Transposed.
//! Used after FM demod to keep the audio band (<= 15 kHz) and reject the
//! 19 kHz stereo pilot, the 38 kHz DSB stereo subcarrier, and SCA carriers
//! further up the MPX baseband (we decode mono only).

use std::f32::consts::TAU;

pub struct AudioLowpass {
    feedforward_0: f32,
    feedforward_1: f32,
    feedforward_2: f32,
    feedback_1: f32,
    feedback_2: f32,
    state_1: f32,
    state_2: f32,
}

impl AudioLowpass {
    pub fn new(sample_rate: u32, cutoff_hz: f32) -> Self {
        let quality_factor = 0.707_f32;
        let omega_0 = TAU * cutoff_hz / sample_rate as f32;
        let cos_omega_0 = omega_0.cos();
        let alpha = omega_0.sin() / (2.0 * quality_factor);
        let normaliser = 1.0 + alpha;
        Self {
            feedforward_0: (1.0 - cos_omega_0) / 2.0 / normaliser,
            feedforward_1: (1.0 - cos_omega_0) / normaliser,
            feedforward_2: (1.0 - cos_omega_0) / 2.0 / normaliser,
            feedback_1: -2.0 * cos_omega_0 / normaliser,
            feedback_2: (1.0 - alpha) / normaliser,
            state_1: 0.0,
            state_2: 0.0,
        }
    }

    pub fn process_in_place(&mut self, audio_samples: &mut [f32]) {
        for sample in audio_samples.iter_mut() {
            let output = self.feedforward_0 * *sample + self.state_1;
            self.state_1 = self.feedforward_1 * *sample - self.feedback_1 * output + self.state_2;
            self.state_2 = self.feedforward_2 * *sample - self.feedback_2 * output;
            *sample = output;
        }
    }
}

#[cfg(test)]
#[path = "audio_lowpass_test.rs"]
mod tests;
