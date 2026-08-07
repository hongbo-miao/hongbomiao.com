use ::fluss::client::FlussConnection;
use ::fluss::error::Result;
use ::fluss::metadata::{DataTypes, Schema, TableDescriptor, TablePath};

use crate::fluss::constants::sensor_table_paths::{
    SENSOR_READINGS_TABLE_PATH, SENSOR_STATUS_TABLE_PATH,
};

/// Creates the log table (append-only history) and the primary-key table
/// (latest-per-sensor lookup) that the demo streams into, idempotently so the
/// publisher can restart without failing on tables that already exist.
pub async fn create_sensor_tables(connection: &FlussConnection) -> Result<()> {
    let admin = connection.get_admin()?;

    let sensor_readings_descriptor = TableDescriptor::builder()
        .schema(
            Schema::builder()
                .column("sensor_id", DataTypes::int())
                .column("location", DataTypes::string())
                .column("temperature_celsius", DataTypes::double())
                .column("pressure_kilopascal", DataTypes::double())
                .column("reading_timestamp", DataTypes::bigint())
                .build()?,
        )
        .build()?;
    admin
        .create_table(
            &TablePath::new(SENSOR_READINGS_TABLE_PATH.0, SENSOR_READINGS_TABLE_PATH.1),
            &sensor_readings_descriptor,
            true,
        )
        .await?;

    let sensor_status_descriptor = TableDescriptor::builder()
        .schema(
            Schema::builder()
                .column("sensor_id", DataTypes::int())
                .column("location", DataTypes::string())
                .column("latest_temperature_celsius", DataTypes::double())
                .column("latest_pressure_kilopascal", DataTypes::double())
                .column("reading_count", DataTypes::bigint())
                .column("updated_timestamp", DataTypes::bigint())
                .primary_key(vec!["sensor_id"])
                .build()?,
        )
        .build()?;
    admin
        .create_table(
            &TablePath::new(SENSOR_STATUS_TABLE_PATH.0, SENSOR_STATUS_TABLE_PATH.1),
            &sensor_status_descriptor,
            true,
        )
        .await?;

    Ok(())
}
