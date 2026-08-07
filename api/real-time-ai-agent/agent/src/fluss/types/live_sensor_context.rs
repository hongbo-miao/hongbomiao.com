use std::collections::BTreeMap;

use crate::fluss::types::sensor_reading_sample::SensorReadingSample;

#[derive(Clone, Debug, Default)]
pub struct LiveSensorContext {
    /// Each sensor's own rolling window, oldest first, most recent last.
    pub readings_by_sensor_id: BTreeMap<i32, Vec<SensorReadingSample>>,
}
