#[cfg(test)]
mod calculate_fused_distance_test {
    use crate::shared::fusion::utils::calculate_fused_distance::calculate_fused_distance;

    #[test]
    fn test_fused_distance_calculation() {
        let lidar_distance = 34.1;
        let radar_distance = 52.4;
        let fused = calculate_fused_distance(lidar_distance, radar_distance);

        // Expected: should be very close to lidar (more accurate sensor)
        // (2500 * 34.1 + 4 * 52.4) / 2504 = 34.14
        println!(
            "lidar={:.2}m, radar={:.2}m, fused={:.2}m",
            lidar_distance, radar_distance, fused
        );

        // Fused should be within 0.1m of lidar distance (not radar)
        assert!((fused - lidar_distance).abs() < 0.1);
        assert!((fused - radar_distance).abs() > 1.0); // Should NOT be close to radar
    }
}
