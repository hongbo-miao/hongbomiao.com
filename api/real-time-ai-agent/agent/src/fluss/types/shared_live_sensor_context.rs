use std::sync::Arc;

use arc_swap::ArcSwap;

use crate::fluss::types::live_sensor_context::LiveSensorContext;

pub type SharedLiveSensorContext = Arc<ArcSwap<LiveSensorContext>>;
