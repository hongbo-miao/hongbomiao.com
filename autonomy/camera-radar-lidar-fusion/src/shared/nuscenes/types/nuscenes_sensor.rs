use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct NuscenesSensor {
    pub token: String,
    pub channel: String,
}
