use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct NuscenesSampleAnnotation {
    #[allow(dead_code)]
    pub token: String,
    pub sample_token: String,
    pub instance_token: String,
    #[allow(dead_code)]
    pub visibility_token: String,
    #[allow(dead_code)]
    pub attribute_tokens: Vec<String>,
    pub translation: [f64; 3],
    pub size: [f64; 3],
    pub rotation: [f64; 4],
    #[allow(dead_code)]
    pub prev: String,
    #[allow(dead_code)]
    pub next: String,
    #[allow(dead_code)]
    pub num_lidar_pts: u32,
    #[allow(dead_code)]
    pub num_radar_pts: u32,
}
