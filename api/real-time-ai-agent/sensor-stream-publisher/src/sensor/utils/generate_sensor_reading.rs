use rand::Rng;

use crate::sensor::types::sensor_reading::SensorReading;
use crate::sensor::utils::current_unix_timestamp_milliseconds::current_unix_timestamp_milliseconds;

/// Sensors 0 and 1 drift into an anomalous temperature range so the agent has
/// something worth flagging when asked which sensors are running hot.
pub fn generate_sensor_reading(sensor_id: i32, location: &str) -> SensorReading {
    let mut random_generator = rand::rng();
    let is_anomalous = sensor_id < 2;
    let temperature_celsius = if is_anomalous {
        random_generator.random_range(75.0..95.0)
    } else {
        random_generator.random_range(18.0..28.0)
    };
    let pressure_kilopascal = random_generator.random_range(98.0..104.0);

    SensorReading {
        sensor_id,
        location: location.to_string(),
        temperature_celsius,
        pressure_kilopascal,
        reading_timestamp: current_unix_timestamp_milliseconds(),
    }
}
