use opencv::core::Scalar;

pub fn calculate_distance_color(distance: f64, max_distance: f64) -> Scalar {
    let ratio = (distance / max_distance).clamp(0.0, 1.0);

    // Gradient from red (close) -> yellow (medium) -> green (far)
    // Red: ratio 0.0
    // Yellow: ratio 0.5
    // Green: ratio 1.0

    let (blue, green, red) = if ratio < 0.5 {
        // Red to Yellow: increase green, keep red high
        let local_ratio = ratio * 2.0;
        (0.0, 255.0 * local_ratio, 255.0)
    } else {
        // Yellow to Green: decrease red, keep green high
        let local_ratio = (ratio - 0.5) * 2.0;
        (0.0, 255.0, 255.0 * (1.0 - local_ratio))
    };

    Scalar::new(blue, green, red, 0.0)
}
