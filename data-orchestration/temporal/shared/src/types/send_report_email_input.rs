use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SendReportEmailInput {
    pub requester_email: String,
    pub report: String,
}
