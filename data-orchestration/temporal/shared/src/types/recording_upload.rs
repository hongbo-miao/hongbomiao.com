use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecordingUpload {
    pub recording_id: String,
    pub requester_email: String,
}
