use super::*;

#[test]
fn boosts_quiet_signal_toward_target() {
    let sample_rate = 48_000;
    let mut agc = Agc::new(sample_rate, 0.25, 50.0);
    let mut buffer: Vec<f32> = (0..sample_rate as usize * 2)
        .map(|i| 0.01 * (i as f32 / sample_rate as f32 * 1000.0 * std::f32::consts::TAU).sin())
        .collect();
    agc.process_in_place(&mut buffer);
    let tail_rms = root_mean_square(&buffer[sample_rate as usize..]);
    assert!(tail_rms > 0.10 && tail_rms < 0.25);
}

#[test]
fn max_gain_ceiling_holds() {
    let sample_rate = 48_000;
    let mut agc = Agc::new(sample_rate, 0.25, 10.0);
    let mut buffer = vec![0.0_f32; sample_rate as usize * 2];
    agc.process_in_place(&mut buffer);
    assert!(agc.gain() <= 10.0 + 1e-3);
}

fn root_mean_square(samples: &[f32]) -> f32 {
    (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt()
}
