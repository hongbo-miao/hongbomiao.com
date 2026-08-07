#[derive(Clone, Debug)]
pub struct SensorReadingSample {
    pub location: String,
    pub temperature_celsius: f64,
    pub pressure_kilopascal: f64,
    pub reading_timestamp: i64,
}
