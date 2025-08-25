use once_cell::sync::Lazy;
use std::collections::HashMap;

#[derive(Clone)]
pub struct PoliceStreamInfo {
    pub name: &'static str,
    pub stream_url: &'static str,
    pub location: &'static str,
}

pub static POLICE_STREAMS: Lazy<HashMap<&'static str, PoliceStreamInfo>> = Lazy::new(|| {
    let mut police_streams_map = HashMap::new();
    police_streams_map.insert(
        "chicago_police_department_zone_08",
        PoliceStreamInfo {
            name: "Chicago Police Department Zone 08",
            stream_url: "https://listen.broadcastify.com/vgb2pt9ms5jr8qc.mp3",
            location: "Chicago, IL",
        },
    );
    police_streams_map
});
