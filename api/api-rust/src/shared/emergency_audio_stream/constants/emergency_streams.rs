use once_cell::sync::Lazy;
use std::collections::HashMap;

#[derive(Clone)]
pub struct FireStreamInfo {
    #[allow(dead_code)]
    pub name: &'static str,
    pub nats_subject: &'static str,
    #[allow(dead_code)]
    pub location: &'static str,
}

pub static EMERGENCY_STREAMS: Lazy<HashMap<&'static str, FireStreamInfo>> = Lazy::new(|| {
    let mut emergency_streams_map = HashMap::new();
    emergency_streams_map.insert(
        "lincoln.fire",
        FireStreamInfo {
            name: "Lincoln Fire",
            nats_subject: "EMERGENCY_AUDIO_STREAMS.lincoln.fire",
            location: "Lincoln, NE",
        },
    );
    emergency_streams_map
});
