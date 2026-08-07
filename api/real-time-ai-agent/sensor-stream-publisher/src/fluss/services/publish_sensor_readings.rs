use std::collections::HashMap;
use std::time::Duration;

use ::fluss::client::FlussConnection;
use ::fluss::error::Result;
use ::fluss::metadata::TablePath;
use ::fluss::row::GenericRow;

use crate::fluss::constants::sensor_table_paths::{
    SENSOR_READINGS_TABLE_PATH, SENSOR_STATUS_TABLE_PATH,
};
use crate::sensor::constants::sensor_locations::SENSOR_LOCATIONS;
use crate::sensor::utils::generate_sensor_reading::generate_sensor_reading;

/// Publishes one reading per sensor per second: an append to the log table
/// for history, and an upsert to the primary-key table for the latest
/// per-sensor snapshot the lookup tool reads.
pub async fn publish_sensor_readings(connection: &FlussConnection) -> Result<()> {
    let sensor_readings_table = connection
        .get_table(&TablePath::new(
            SENSOR_READINGS_TABLE_PATH.0,
            SENSOR_READINGS_TABLE_PATH.1,
        ))
        .await?;
    let sensor_status_table = connection
        .get_table(&TablePath::new(
            SENSOR_STATUS_TABLE_PATH.0,
            SENSOR_STATUS_TABLE_PATH.1,
        ))
        .await?;

    let append_writer = sensor_readings_table.new_append()?.create_writer()?;
    let upsert_writer = sensor_status_table.new_upsert()?.create_writer()?;

    let mut reading_counts_by_sensor: HashMap<i32, i64> = HashMap::new();

    loop {
        for (sensor_id, location) in (0..SENSOR_LOCATIONS.len() as i32).zip(SENSOR_LOCATIONS) {
            let reading = generate_sensor_reading(sensor_id, location);
            let reading_count = reading_counts_by_sensor.entry(sensor_id).or_insert(0);
            *reading_count += 1;

            let mut readings_row = GenericRow::new(5);
            readings_row.set_field(0, reading.sensor_id);
            readings_row.set_field(1, reading.location.as_str());
            readings_row.set_field(2, reading.temperature_celsius);
            readings_row.set_field(3, reading.pressure_kilopascal);
            readings_row.set_field(4, reading.reading_timestamp);
            append_writer.append(&readings_row)?;

            let mut status_row = GenericRow::new(6);
            status_row.set_field(0, reading.sensor_id);
            status_row.set_field(1, reading.location.as_str());
            status_row.set_field(2, reading.temperature_celsius);
            status_row.set_field(3, reading.pressure_kilopascal);
            status_row.set_field(4, *reading_count);
            status_row.set_field(5, reading.reading_timestamp);
            upsert_writer.upsert(&status_row)?;
        }

        append_writer.flush().await?;
        upsert_writer.flush().await?;
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}
