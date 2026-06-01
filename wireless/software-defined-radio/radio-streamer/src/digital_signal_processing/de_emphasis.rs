//! FM de-emphasis: a single-pole IIR low-pass that inverts the transmitter's
//! pre-emphasis (a high-frequency boost added at TX to push hiss down on the
//! receive side). Time constant tau picks the corner:
//!   alpha = exp(-1 / (tau * sample_rate))
//!   y[n] = (1 - alpha) * x[n] + alpha * y[n-1]
//! 75 us (Americas/Korea, -3 dB at 2.12 kHz) or 50 us (most of the world, 3.18 kHz).

pub struct DeEmphasis {
    feedback_coefficient: f32,
    last_output: f32,
}

impl DeEmphasis {
    pub fn new(sample_rate: u32, time_constant_us: u32) -> Self {
        let tau_seconds = time_constant_us as f32 * 1e-6;
        let feedback_coefficient = (-1.0 / (tau_seconds * sample_rate as f32)).exp();
        Self {
            feedback_coefficient,
            last_output: 0.0,
        }
    }

    pub fn process_in_place(&mut self, audio_samples: &mut [f32]) {
        let feedforward_coefficient = 1.0 - self.feedback_coefficient;
        for sample in audio_samples.iter_mut() {
            let output =
                feedforward_coefficient * *sample + self.feedback_coefficient * self.last_output;
            self.last_output = output;
            *sample = output;
        }
    }
}

#[cfg(test)]
#[path = "de_emphasis_test.rs"]
mod tests;
