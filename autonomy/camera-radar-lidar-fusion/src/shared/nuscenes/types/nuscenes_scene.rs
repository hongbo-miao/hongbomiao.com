use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct NuscenesScene {
    pub first_sample_token: String,
    pub name: String,
    pub log_token: String,
}
