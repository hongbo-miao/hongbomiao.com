use num_complex::Complex;

use super::*;

#[test]
fn pure_carrier_demodulates_to_zero() {
    let mut demodulator = FmDemodulator::new(240_000, 75_000);
    let iq_samples: Vec<Complex<f32>> = (0..10_000).map(|_| Complex::new(0.5, 0.0)).collect();
    let mut audio_out = Vec::new();
    demodulator.process(&iq_samples, &mut audio_out);
    let tail_mean: f32 = audio_out[1_000..].iter().sum::<f32>() / (audio_out.len() - 1_000) as f32;
    assert!(tail_mean.abs() < 1e-4);
}

#[test]
fn full_deviation_offset_demodulates_to_unity() {
    let sample_rate = 240_000_u32;
    let peak_deviation_hz = 75_000_u32;
    let mut demodulator = FmDemodulator::new(sample_rate, peak_deviation_hz);
    let omega = std::f32::consts::TAU * peak_deviation_hz as f32 / sample_rate as f32;
    let iq_samples: Vec<Complex<f32>> = (0..10_000)
        .map(|index| {
            let phase = omega * index as f32;
            Complex::new(phase.cos(), phase.sin())
        })
        .collect();
    let mut audio_out = Vec::new();
    demodulator.process(&iq_samples, &mut audio_out);
    let tail_mean: f32 = audio_out[100..].iter().sum::<f32>() / (audio_out.len() - 100) as f32;
    assert!((tail_mean - 1.0).abs() < 0.01);
}

#[test]
fn modulated_tone_recovers() {
    let sample_rate = 240_000_u32;
    let peak_deviation_hz = 75_000_u32;
    let tone_hz = 1_000.0_f32;
    let mut demodulator = FmDemodulator::new(sample_rate, peak_deviation_hz);
    let mut phase = 0.0_f32;
    let iq_samples: Vec<Complex<f32>> = (0..sample_rate as usize / 10)
        .map(|index| {
            let time = index as f32 / sample_rate as f32;
            let instantaneous_hz =
                peak_deviation_hz as f32 * (std::f32::consts::TAU * tone_hz * time).sin();
            phase += std::f32::consts::TAU * instantaneous_hz / sample_rate as f32;
            Complex::new(phase.cos(), phase.sin())
        })
        .collect();
    let mut audio_out = Vec::new();
    demodulator.process(&iq_samples, &mut audio_out);
    let root_mean_square = (audio_out[200..].iter().map(|s| s * s).sum::<f32>()
        / (audio_out.len() - 200) as f32)
        .sqrt();
    assert!(root_mean_square > 0.5);
}
