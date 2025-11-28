pub fn get_nuscenes_location_reference_gps_coordinate(location: &str) -> Option<(f64, f64)> {
    match location {
        // Southwest corner of each location
        "singapore-onenorth" => Some((1.2882100868743724, 103.78475189208984)),
        "singapore-hollandvillage" => Some((1.2993652317780957, 103.78217697143555)),
        "singapore-queenstown" => Some((1.2782562240223188, 103.76741409301758)),
        "boston-seaport" => Some((42.336849169438615, -71.05785369873047)),
        _ => None,
    }
}
