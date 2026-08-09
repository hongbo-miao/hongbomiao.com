use std::time::{SystemTime, UNIX_EPOCH};

const SHORT_UUID_LENGTH: usize = 8;

// Egress writes a playlist, its segments, and (optionally) a manifest for a single
// session. Grouping them under one prefix lets a single S3 lifecycle rule expire an
// entire session as a unit instead of trying to match by file suffix, which S3
// lifecycle rules cannot do.
pub fn build_session_prefix(room_name: &str) -> String {
    let unix_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    let short_uuid = &uuid::Uuid::new_v4().to_string()[..SHORT_UUID_LENGTH];

    format!("stream/{room_name}/{unix_seconds}-{short_uuid}")
}
