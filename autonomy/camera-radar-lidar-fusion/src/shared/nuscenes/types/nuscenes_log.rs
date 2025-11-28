use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct NuscenesLog {
    pub token: String,
    pub location: String,
}
