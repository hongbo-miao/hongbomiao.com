use std::f32::consts::TAU;

use super::*;

fn root_mean_square(samples: &[f32]) -> f32 {
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

#[test]
fn identity_passes_through() {
    let mut resampler = RationalResampler::new(1, 1);
    let input: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let mut out = Vec::new();
    resampler.process(&input, &mut out);
    assert_eq!(out, input);
}

#[test]
fn output_count_tracks_ratio() {
    let mut resampler = RationalResampler::new(3, 5);
    let input = vec![0.0_f32; 5000];
    let mut out = Vec::new();
    resampler.process(&input, &mut out);
    let expected = 5000 * 3 / 5;
    assert!((out.len() as i64 - expected as i64).abs() <= 2);
}

#[test]
fn preserves_in_band_tone_amplitude() {
    let input_sample_rate = 80_000.0_f32;
    let mut resampler = RationalResampler::new(3, 5);
    let input: Vec<f32> = (0..80_000)
        .map(|n| (TAU * 1000.0 * n as f32 / input_sample_rate).sin())
        .collect();
    let mut out = Vec::new();
    resampler.process(&input, &mut out);
    let steady = &out[2000..];
    let input_rms = root_mean_square(&input);
    let output_rms = root_mean_square(steady);
    assert!((output_rms - input_rms).abs() < 0.05);
}
