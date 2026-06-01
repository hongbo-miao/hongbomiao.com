use std::f32::consts::TAU;

use super::*;

fn make_tone(sample_rate: u32, hz: f32, count: usize) -> Vec<f32> {
    (0..count)
        .map(|index| {
            let time = index as f32 / sample_rate as f32;
            (TAU * hz * time).sin()
        })
        .collect()
}

fn root_mean_square(samples: &[f32]) -> f32 {
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

#[test]
fn passes_audio_band() {
    let sample_rate = 240_000_u32;
    let mut filter = AudioLowpass::new(sample_rate, 15_000.0);
    let mut tone = make_tone(sample_rate, 1_000.0, 48_000);
    filter.process_in_place(&mut tone);
    assert!(root_mean_square(&tone[16_000..]) > 0.5);
}

#[test]
fn rejects_19khz_pilot() {
    let sample_rate = 240_000_u32;
    let mut filter = AudioLowpass::new(sample_rate, 15_000.0);
    let mut pilot = make_tone(sample_rate, 19_000.0, 48_000);
    filter.process_in_place(&mut pilot);
    let input_rms = root_mean_square(&make_tone(sample_rate, 19_000.0, 48_000));
    assert!(root_mean_square(&pilot[16_000..]) < 0.7 * input_rms);
}
