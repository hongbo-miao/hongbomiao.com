pub struct SensorReading {
    pub sensor_id: i32,
    pub location: String,
    pub temperature_celsius: f64,
    pub pressure_kilopascal: f64,
    pub reading_timestamp: i64,
}
