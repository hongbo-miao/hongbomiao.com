use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct TelemetryRecord {
    pub publisher_id: String,
    pub timestamp_ns: i64,
    pub temperature_c: Option<f64>,
    pub humidity_pct: Option<f64>,
}
