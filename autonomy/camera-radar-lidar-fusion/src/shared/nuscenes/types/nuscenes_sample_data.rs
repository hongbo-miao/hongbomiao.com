use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct NuscenesSampleData {
    #[allow(dead_code)]
    pub token: String,
    pub sample_token: String,
    pub ego_pose_token: String,
    pub calibrated_sensor_token: String,
    pub filename: String,
    pub is_key_frame: bool,
    #[allow(dead_code)]
    pub width: u32,
    #[allow(dead_code)]
    pub height: u32,
}
