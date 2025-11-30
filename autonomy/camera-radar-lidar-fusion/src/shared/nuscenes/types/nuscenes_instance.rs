use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct NuscenesInstance {
    pub token: String,
    pub category_token: String,
    #[allow(dead_code)]
    pub nbr_annotations: u32,
    #[allow(dead_code)]
    pub first_annotation_token: String,
    #[allow(dead_code)]
    pub last_annotation_token: String,
}
