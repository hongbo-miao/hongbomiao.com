use std::sync::Arc;
use std::time::Duration;

use ::fluss::client::FlussConnection;
use ::fluss::error::Result;
use ::fluss::metadata::TablePath;
use ::fluss::row::InternalRow;
use ::fluss::rpc::message::OffsetSpec;

use crate::fluss::constants::live_sensor_context_window_size::LIVE_SENSOR_CONTEXT_WINDOW_SIZE_PER_SENSOR;
use crate::fluss::types::sensor_reading_sample::SensorReadingSample;
use crate::fluss::types::shared_live_sensor_context::SharedLiveSensorContext;

/// Streams the log table's tail into the shared rolling window forever, so
/// every agent turn sees readings that are at most a couple of seconds old.
pub async fn tail_sensor_readings(
    connection: &FlussConnection,
    sensor_readings_table_path: &TablePath,
    shared_live_sensor_context: SharedLiveSensorContext,
) -> Result<()> {
    let table = connection.get_table(sensor_readings_table_path).await?;
    let admin = connection.get_admin()?;

    let latest_offsets = admin
        .list_offsets(sensor_readings_table_path, &[0], OffsetSpec::Latest)
        .await?;
    let latest_offset = latest_offsets[&0];

    let log_scanner = table.new_scan().create_log_scanner()?;
    log_scanner.subscribe(0, latest_offset).await?;

    loop {
        let scan_records = log_scanner.poll(Duration::from_secs(2)).await?;
        if scan_records.is_empty() {
            continue;
        }

        let mut live_sensor_context = (**shared_live_sensor_context.load()).clone();
        for record in scan_records {
            let row = record.row();
            let sensor_id = row.get_int(0)?;
            let sensor_window = live_sensor_context
                .readings_by_sensor_id
                .entry(sensor_id)
                .or_default();
            sensor_window.push(SensorReadingSample {
                location: row.get_string(1)?.to_string(),
                temperature_celsius: row.get_double(2)?,
                pressure_kilopascal: row.get_double(3)?,
                reading_timestamp: row.get_long(4)?,
            });
            let overflow = sensor_window
                .len()
                .saturating_sub(LIVE_SENSOR_CONTEXT_WINDOW_SIZE_PER_SENSOR);
            if overflow > 0 {
                sensor_window.drain(0..overflow);
            }
        }

        shared_live_sensor_context.store(Arc::new(live_sensor_context));
    }
}
