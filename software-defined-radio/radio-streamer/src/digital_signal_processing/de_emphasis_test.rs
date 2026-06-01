use super::*;

#[test]
fn dc_passes_with_unity_gain() {
    let mut filter = DeEmphasis::new(240_000, 75);
    let mut buffer = vec![1.0_f32; 240_000];
    filter.process_in_place(&mut buffer);
    let tail_mean: f32 = buffer[200_000..].iter().sum::<f32>() / 40_000.0;
    assert!((tail_mean - 1.0).abs() < 1e-3);
}

#[test]
fn attenuates_high_frequencies() {
    let sample_rate = 240_000_u32;
    let make_tone = |hz: f32| -> Vec<f32> {
        (0..sample_rate as usize)
            .map(|index| {
                let time = index as f32 / sample_rate as f32;
                (std::f32::consts::TAU * hz * time).sin()
            })
            .collect()
    };
    let root_mean_square = |samples: &[f32]| -> f32 {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    };

    let mut low = make_tone(1_000.0);
    let mut high = make_tone(10_000.0);
    DeEmphasis::new(sample_rate, 75).process_in_place(&mut low);
    DeEmphasis::new(sample_rate, 75).process_in_place(&mut high);

    assert!(root_mean_square(&high[10_000..]) < 0.3 * root_mean_square(&low[10_000..]));
}
