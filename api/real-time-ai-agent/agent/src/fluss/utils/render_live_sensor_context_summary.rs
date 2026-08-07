use crate::fluss::types::live_sensor_context::LiveSensorContext;

struct SensorAggregate {
    location: String,
    latest_temperature_celsius: f64,
    latest_pressure_kilopascal: f64,
    minimum_temperature_celsius: f64,
    maximum_temperature_celsius: f64,
    average_temperature_celsius: f64,
    latest_reading_timestamp: i64,
    sample_count: usize,
}

/// Renders each sensor's own rolling window into a plain-text summary
/// suitable for injection as agent context on every turn.
pub fn render_live_sensor_context_summary(live_sensor_context: &LiveSensorContext) -> String {
    if live_sensor_context.readings_by_sensor_id.is_empty() {
        return "No sensor readings have streamed in yet.".to_string();
    }

    let mut total_sample_count = 0;
    let mut summary_lines = Vec::new();
    for (sensor_id, sensor_window) in &live_sensor_context.readings_by_sensor_id {
        let Some(latest_reading) = sensor_window.last() else {
            continue;
        };

        let sample_count = sensor_window.len();
        let temperature_sum: f64 = sensor_window
            .iter()
            .map(|reading| reading.temperature_celsius)
            .sum();
        let aggregate = SensorAggregate {
            location: latest_reading.location.clone(),
            latest_temperature_celsius: latest_reading.temperature_celsius,
            latest_pressure_kilopascal: latest_reading.pressure_kilopascal,
            minimum_temperature_celsius: sensor_window
                .iter()
                .map(|reading| reading.temperature_celsius)
                .fold(f64::INFINITY, f64::min),
            maximum_temperature_celsius: sensor_window
                .iter()
                .map(|reading| reading.temperature_celsius)
                .fold(f64::NEG_INFINITY, f64::max),
            average_temperature_celsius: temperature_sum / sample_count as f64,
            latest_reading_timestamp: latest_reading.reading_timestamp,
            sample_count,
        };
        total_sample_count += sample_count;

        summary_lines.push(format!(
            "- sensor {sensor_id} ({}): latest={:.1}C ({:.1}kPa) at unix_ms={}, avg={:.1}C, min={:.1}C, max={:.1}C, samples={}",
            aggregate.location,
            aggregate.latest_temperature_celsius,
            aggregate.latest_pressure_kilopascal,
            aggregate.latest_reading_timestamp,
            aggregate.average_temperature_celsius,
            aggregate.minimum_temperature_celsius,
            aggregate.maximum_temperature_celsius,
            aggregate.sample_count,
        ));
    }

    summary_lines.insert(
        0,
        format!("Live sensor readings ({total_sample_count} samples across each sensor's own rolling window):"),
    );
    summary_lines.join("\n")
}
