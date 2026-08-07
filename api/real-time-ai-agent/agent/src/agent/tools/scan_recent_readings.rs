use std::sync::Arc;
use std::time::Duration;

use ::fluss::client::{FlussAdmin, FlussTable};
use ::fluss::metadata::TablePath;
use ::fluss::row::InternalRow;
use ::fluss::rpc::message::OffsetSpec;
use rig::tool::{Tool, ToolContext};
use serde::Deserialize;
use serde_json::{Value, json};

const MAXIMUM_REQUESTED_READING_COUNT: usize = 100;
const SCAN_POLL_TIMEOUT: Duration = Duration::from_secs(3);

#[derive(Debug, thiserror::Error)]
pub enum ScanRecentReadingsError {
    #[error("Fluss scan failed: {0}")]
    Fluss(#[from] fluss::error::Error),
}

#[derive(Deserialize)]
pub struct ScanRecentReadingsArgs {
    /// Number of most recent raw readings to return, across all sensors.
    pub reading_count: usize,
}

/// Bounded scan of the `sensor_readings` log table starting a fixed number of
/// records behind the current tail - the Streamhouse diagram's "batch reads"
/// pattern (a bounded snapshot scan, as opposed to an open-ended tail),
/// exposed as an agent tool.
pub struct ScanRecentReadings {
    sensor_readings_table: FlussTable<'static>,
    sensor_readings_admin: Arc<FlussAdmin>,
    sensor_readings_table_path: TablePath,
}

impl ScanRecentReadings {
    pub fn new(
        sensor_readings_table: FlussTable<'static>,
        sensor_readings_admin: Arc<FlussAdmin>,
        sensor_readings_table_path: TablePath,
    ) -> Self {
        Self {
            sensor_readings_table,
            sensor_readings_admin,
            sensor_readings_table_path,
        }
    }
}

impl Tool for ScanRecentReadings {
    const NAME: &'static str = "scan_recent_readings";
    type Error = ScanRecentReadingsError;
    type Args = ScanRecentReadingsArgs;
    type Output = Value;

    fn description(&self) -> String {
        "Returns the most recent raw sensor readings from the sensor_readings \
        log table, unaggregated. Use this when the user wants to see actual \
        recent data points rather than a summary."
            .to_string()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "reading_count": {
                    "type": "integer",
                    "description": "How many recent readings to return, e.g. 10"
                }
            },
            "required": ["reading_count"]
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let requested_reading_count = args.reading_count.min(MAXIMUM_REQUESTED_READING_COUNT);

        let latest_offsets = self
            .sensor_readings_admin
            .list_offsets(&self.sensor_readings_table_path, &[0], OffsetSpec::Latest)
            .await?;
        let latest_offset = latest_offsets[&0];
        let start_offset = (latest_offset - requested_reading_count as i64).max(0);

        let log_scanner = self.sensor_readings_table.new_scan().create_log_scanner()?;
        log_scanner.subscribe(0, start_offset).await?;

        let mut readings = Vec::with_capacity(requested_reading_count);
        while readings.len() < requested_reading_count {
            let scan_records = log_scanner.poll(SCAN_POLL_TIMEOUT).await?;
            if scan_records.is_empty() {
                break;
            }
            for record in scan_records {
                let row = record.row();
                readings.push(json!({
                    "sensor_id": row.get_int(0)?,
                    "location": row.get_string(1)?,
                    "temperature_celsius": row.get_double(2)?,
                    "pressure_kilopascal": row.get_double(3)?,
                    "reading_timestamp": row.get_long(4)?,
                    "offset": record.offset(),
                }));
                if readings.len() >= requested_reading_count {
                    break;
                }
            }
        }

        Ok(json!({ "readings": readings }))
    }
}
