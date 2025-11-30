use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct NuscenesCategory {
    pub token: String,
    pub name: String,
    #[allow(dead_code)]
    pub description: String,
}
