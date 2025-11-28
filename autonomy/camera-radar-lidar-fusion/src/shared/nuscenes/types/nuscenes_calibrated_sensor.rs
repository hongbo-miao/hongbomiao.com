use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct NuscenesCalibratedSensor {
    pub token: String,
    pub sensor_token: String,
    pub rotation: [f64; 4],
    pub translation: [f64; 3],
    #[serde(default)]
    pub camera_intrinsic: Vec<Vec<f64>>,
}
