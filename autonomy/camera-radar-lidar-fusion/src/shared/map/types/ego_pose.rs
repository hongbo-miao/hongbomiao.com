use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct EgoPose {
    pub token: String,
    #[allow(dead_code)]
    pub timestamp: u64,
    #[allow(dead_code)]
    pub rotation: [f64; 4],
    pub translation: [f64; 3],
}
