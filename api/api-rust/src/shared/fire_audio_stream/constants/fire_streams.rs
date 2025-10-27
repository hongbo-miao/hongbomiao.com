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

pub static FIRE_STREAMS: Lazy<HashMap<&'static str, FireStreamInfo>> = Lazy::new(|| {
    let mut fire_streams_map = HashMap::new();
    fire_streams_map.insert(
        "lincoln_fire",
        FireStreamInfo {
            name: "Lincoln Fire",
            nats_subject: "FIRE_AUDIO_STREAMS.lincoln_fire",
            location: "Lincoln, NE",
        },
    );
    fire_streams_map
});
