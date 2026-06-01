use std::f32::consts::TAU;

use num_complex::Complex;

use super::*;

#[test]
fn fir_has_unity_dc_gain() {
    let taps = design_lowpass_fir(257, 12_000.0, 2_400_000.0);
    let dc: f32 = taps.iter().sum();
    assert!((dc - 1.0).abs() < 1e-5);
}

#[test]
fn ddc_decimates_by_correct_factor() {
    let mut ddc = Ddc::new(2_400_000, 48_000, 0.0, 12_000.0);
    let chunk: Vec<Complex<f32>> = (0..2_400).map(|_| Complex::new(0.0, 0.0)).collect();
    let mut out = Vec::new();
    for _ in 0..3 {
        ddc.process(&chunk, &mut out);
    }
    assert_eq!(out.len(), 144);
}

#[test]
fn ddc_shifts_tone_to_dc() {
    let input_rate = 2_400_000_u32;
    let output_rate = 48_000_u32;
    let f_tone = 100_000.0_f64;

    let mut ddc = Ddc::new(input_rate, output_rate, f_tone, 12_000.0);

    let omega = TAU * f_tone as f32 / input_rate as f32;
    let chunk: Vec<Complex<f32>> = (0..24_000)
        .map(|i| {
            let theta = omega * i as f32;
            let (s, c) = theta.sin_cos();
            Complex::new(c, s)
        })
        .collect();

    let mut out = Vec::new();
    ddc.process(&chunk, &mut out);

    let tail = &out[out.len() / 2..];
    for &y in tail {
        assert!((y.norm() - 1.0).abs() < 1e-2);
    }
}
