//! Audio AGC: track the input envelope with an asymmetric one-pole filter
//! (fast attack, slow release so it catches transients but doesn't pump during
//! brief quiet passages), pick a gain that lands the envelope on
//! `target_envelope`, then smooth the gain itself with a separate pole so the
//! gain doesn't audibly bounce sample to sample. `max_gain` caps the boost so
//! silence is not amplified to the noise ceiling.

pub struct Agc {
    target_envelope: f32,
    max_gain: f32,
    envelope: f32,
    gain: f32,
    attack_alpha: f32,
    release_alpha: f32,
    gain_smoothing_alpha: f32,
}

impl Agc {
    pub fn new(sample_rate: u32, target_envelope: f32, max_gain: f32) -> Self {
        let attack_tau_seconds = 0.010_f32;
        let release_tau_seconds = 0.300_f32;
        let gain_smoothing_tau_seconds = 0.030_f32;
        let sample_rate = sample_rate as f32;
        Self {
            target_envelope,
            max_gain,
            envelope: target_envelope,
            gain: 1.0,
            attack_alpha: (-1.0 / (attack_tau_seconds * sample_rate)).exp(),
            release_alpha: (-1.0 / (release_tau_seconds * sample_rate)).exp(),
            gain_smoothing_alpha: (-1.0 / (gain_smoothing_tau_seconds * sample_rate)).exp(),
        }
    }

    #[allow(dead_code)]
    pub fn gain(&self) -> f32 {
        self.gain
    }

    pub fn process_in_place(&mut self, audio_samples: &mut [f32]) {
        for sample in audio_samples.iter_mut() {
            let magnitude = sample.abs();
            let envelope_alpha = if magnitude > self.envelope {
                self.attack_alpha
            } else {
                self.release_alpha
            };
            self.envelope = envelope_alpha * self.envelope + (1.0 - envelope_alpha) * magnitude;

            let target_gain =
                (self.target_envelope / self.envelope.max(1e-6)).clamp(0.1, self.max_gain);
            self.gain = self.gain_smoothing_alpha * self.gain
                + (1.0 - self.gain_smoothing_alpha) * target_gain;

            *sample *= self.gain;
        }
    }
}

#[cfg(test)]
#[path = "agc_test.rs"]
mod tests;
