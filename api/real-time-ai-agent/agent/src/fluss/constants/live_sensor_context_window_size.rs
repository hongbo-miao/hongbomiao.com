/// Maximum number of recent readings kept per sensor in the rolling window
/// that gets rendered into the agent's per-turn context. Windowing per
/// sensor (rather than one shared window across the whole fleet) keeps a
/// fast-publishing sensor from evicting a slow or stalled one out of context.
pub const LIVE_SENSOR_CONTEXT_WINDOW_SIZE_PER_SENSOR: usize = 10;
