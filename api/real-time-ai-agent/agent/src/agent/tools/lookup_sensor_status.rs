use ::fluss::client::FlussTable;
use ::fluss::row::{GenericRow, InternalRow};
use rig::tool::{Tool, ToolContext};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Debug, thiserror::Error)]
pub enum LookupSensorStatusError {
    #[error("Fluss lookup failed: {0}")]
    Fluss(#[from] fluss::error::Error),
}

#[derive(Deserialize)]
pub struct LookupSensorStatusArgs {
    pub sensor_id: i32,
}

/// Point lookup on the primary-key `sensor_status` table - the Streamhouse
/// diagram's "lookup join" read pattern, exposed as an agent tool.
pub struct LookupSensorStatus {
    sensor_status_table: FlussTable<'static>,
}

impl LookupSensorStatus {
    pub fn new(sensor_status_table: FlussTable<'static>) -> Self {
        Self {
            sensor_status_table,
        }
    }
}

impl Tool for LookupSensorStatus {
    const NAME: &'static str = "lookup_sensor_status";
    type Error = LookupSensorStatusError;
    type Args = LookupSensorStatusArgs;
    type Output = Value;

    fn description(&self) -> String {
        "Looks up the exact latest reading for one sensor by its sensor_id, by \
        primary-key lookup on the sensor_status table. Use this when the user \
        asks about a specific sensor rather than the fleet in general."
            .to_string()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "sensor_id": {
                    "type": "integer",
                    "description": "The sensor_id to look up, e.g. 3"
                }
            },
            "required": ["sensor_id"]
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let mut lookup_key = GenericRow::new(1);
        lookup_key.set_field(0, args.sensor_id);

        let mut lookuper = self.sensor_status_table.new_lookup()?.create_lookuper()?;
        let result = lookuper.lookup(&lookup_key).await?;

        let Some(row) = result.get_single_row()? else {
            return Ok(json!({ "found": false, "sensor_id": args.sensor_id }));
        };

        Ok(json!({
            "found": true,
            "sensor_id": row.get_int(0)?,
            "location": row.get_string(1)?,
            "latest_temperature_celsius": row.get_double(2)?,
            "latest_pressure_kilopascal": row.get_double(3)?,
            "reading_count": row.get_long(4)?,
            "updated_timestamp": row.get_long(5)?,
        }))
    }
}
