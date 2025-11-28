use crate::shared::map::utils::get_reference_coordinate::get_nuscenes_location_reference_gps_coordinate;

// Earth radius in meters
const EARTH_RADIUS_METERS: f64 = 6_378_137.0;

fn get_coordinate(
    reference_latitude: f64,
    reference_longitude: f64,
    bearing_rad: f64,
    distance_m: f64,
) -> (f64, f64) {
    let latitude_radians = reference_latitude.to_radians();
    let longitude_radians = reference_longitude.to_radians();
    let angular_distance = distance_m / EARTH_RADIUS_METERS;

    let target_latitude_radians = (latitude_radians.sin() * angular_distance.cos()
        + latitude_radians.cos() * angular_distance.sin() * bearing_rad.cos())
    .asin();

    let target_longitude_radians = longitude_radians
        + (bearing_rad.sin() * angular_distance.sin() * latitude_radians.cos())
            .atan2(angular_distance.cos() - latitude_radians.sin() * target_latitude_radians.sin());

    (
        target_latitude_radians.to_degrees(),
        target_longitude_radians.to_degrees(),
    )
}

pub fn derive_latitude_longitude(location: &str, x: f64, y: f64) -> Option<(f64, f64)> {
    let (reference_latitude, reference_longitude) =
        get_nuscenes_location_reference_gps_coordinate(location)?;

    // Compute bearing (radians) and distance (meters) from the local x, y offsets relative to the reference coordinate
    let bearing_rad = x.atan2(y);
    let distance_m = x.hypot(y);

    Some(get_coordinate(
        reference_latitude,
        reference_longitude,
        bearing_rad,
        distance_m,
    ))
}
