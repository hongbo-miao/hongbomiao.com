/// Calculates the fused distance using variance-weighted averaging (inverse variance weighting).
/// This is a common approach in sensor fusion that gives more weight to more accurate sensors.
///
/// # Sensor Characteristics (typical values):
/// - Lidar: High accuracy, ~2cm standard deviation
/// - Radar: Lower accuracy, ~50cm standard deviation
///
/// # Formula:
/// fused_distance = (w_lidar * d_lidar + w_radar * d_radar) / (w_lidar + w_radar)
/// where w_i = 1 / variance_i
///
/// # Arguments:
/// * `lidar_distance` - Distance measurement from lidar sensor
/// * `radar_distance` - Distance measurement from radar sensor
///
/// # Returns:
/// Fused distance that optimally combines both measurements
pub fn calculate_fused_distance(lidar_distance: f64, radar_distance: f64) -> f64 {
    // Sensor noise characteristics (standard deviations in meters)
    const LIDAR_STD_DEV: f64 = 0.02; // Lidar: ~2cm accuracy
    const RADAR_STD_DEV: f64 = 0.5; // Radar: ~50cm accuracy

    // Calculate variances (sigma^2)
    let lidar_variance = LIDAR_STD_DEV * LIDAR_STD_DEV;
    let radar_variance = RADAR_STD_DEV * RADAR_STD_DEV;

    // Calculate weights (inverse variance)
    let lidar_weight = 1.0 / lidar_variance;
    let radar_weight = 1.0 / radar_variance;

    // Weighted average
    (lidar_weight * lidar_distance + radar_weight * radar_distance) / (lidar_weight + radar_weight)
}

#[cfg(test)]
#[path = "calculate_fused_distance_test.rs"]
mod tests;
