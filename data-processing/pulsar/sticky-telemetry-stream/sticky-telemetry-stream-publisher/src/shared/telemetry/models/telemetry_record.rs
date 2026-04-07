use apache_avro::AvroSchema;
use pulsar::SerializeMessage;
use pulsar::producer::Message;
use serde::Serialize;

#[derive(Debug, Serialize, AvroSchema)]
pub struct TelemetryRecord {
    pub publisher_id: String,
    pub timestamp_ns: i64,
    pub temperature_c: Option<f64>,
    pub humidity_pct: Option<f64>,
}

impl SerializeMessage for TelemetryRecord {
    fn serialize_message(input: Self) -> Result<Message, pulsar::Error> {
        let schema = Self::get_schema();
        let partition_key = Some(input.publisher_id.clone());
        let avro_value = apache_avro::to_value(&input).map_err(|error| {
            pulsar::Error::Custom(format!("Failed to convert to Avro value: {error}"))
        })?;
        let payload = apache_avro::to_avro_datum(&schema, avro_value).map_err(|error| {
            pulsar::Error::Custom(format!("Failed to serialize Avro datum: {error}"))
        })?;
        Ok(Message {
            payload,
            partition_key,
            ..Default::default()
        })
    }
}
