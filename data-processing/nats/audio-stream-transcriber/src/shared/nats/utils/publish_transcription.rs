use crate::shared::speaches::types::transcription_response::TranscriptionResponse;
use crate::transcript_capnp;
use anyhow::Result;
use async_nats::header::{HeaderName, HeaderValue};
use async_nats::jetstream;
use tracing::{error, info};
use uuid::Uuid;

pub async fn publish_transcription(
    jetstream_context: &jetstream::Context,
    subject: &str,
    stream_id: &str,
    timestamp_ns: i64,
    transcription_response: &TranscriptionResponse,
    segment_start_s: f64,
    segment_end_s: f64,
) -> Result<()> {
    // Build and serialize the Cap'n Proto message completely before any async operations
    let transcript_bytes = {
        let mut message = capnp::message::Builder::new_default();
        {
            let mut transcript = message.init_root::<transcript_capnp::transcript::Builder>();
            transcript.set_stream_id(stream_id);
            transcript.set_timestamp_ns(timestamp_ns);
            transcript.set_text(&transcription_response.text);
            transcript.set_language(&transcription_response.language);
            transcript.set_duration_s(transcription_response.duration);
            transcript.set_segment_start_s(segment_start_s);
            transcript.set_segment_end_s(segment_end_s);

            // Build words list
            let word_count = transcription_response.words.len() as u32;
            let mut words_builder = transcript.init_words(word_count);
            for (index, transcription_word) in transcription_response.words.iter().enumerate() {
                let mut word_builder = words_builder.reborrow().get(index as u32);
                word_builder.set_word(&transcription_word.word);
                word_builder.set_start_s(transcription_word.start);
                word_builder.set_end_s(transcription_word.end);
                word_builder.set_probability(transcription_word.probability);
            }
        }
        let mut bytes = Vec::new();
        capnp::serialize::write_message(&mut bytes, &message)?;
        bytes
    };

    let mut headers = async_nats::HeaderMap::new();
    headers.insert(
        HeaderName::from_static("Nats-Msg-Id"),
        HeaderValue::from(Uuid::new_v4().to_string()),
    );

    match jetstream_context
        .publish_with_headers(subject.to_string(), headers, transcript_bytes.into())
        .await
    {
        Ok(_acknowledgement) => {
            info!(
                "Published transcription to NATS (subject={subject}, stream_id={stream_id}): {}",
                transcription_response.text
            );
            Ok(())
        }
        Err(error) => {
            error!(
                "Failed to publish transcription to NATS (subject={subject}, stream_id={stream_id}): {error}",
            );
            Err(error.into())
        }
    }
}
