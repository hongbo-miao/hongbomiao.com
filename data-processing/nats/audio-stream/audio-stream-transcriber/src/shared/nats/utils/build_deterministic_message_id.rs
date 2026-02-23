use uuid::Uuid;

const TRANSCRIPTION_NAMESPACE: Uuid = Uuid::NAMESPACE_URL;

pub fn build_deterministic_message_id(stream_id: &str, timestamp_ns: i64) -> String {
    let content = format!("{stream_id}:{timestamp_ns}");
    Uuid::new_v5(&TRANSCRIPTION_NAMESPACE, content.as_bytes()).to_string()
}
