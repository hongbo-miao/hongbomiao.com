use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct NuscenesSample {
    pub token: String,
    pub next: String,
    #[allow(dead_code)]
    pub prev: String,
    #[allow(dead_code)]
    pub scene_token: String,
}
