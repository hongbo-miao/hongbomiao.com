use crate::shared::postgres::services::write_transcription_to_postgres::write_transcription_to_postgres;
use crate::shared::transcripts::types::transcript::{Transcript, TranscriptWord};
use crate::transcription_capnp;
use anyhow::{Context, Result};
use sqlx::PgPool;

pub async fn process_message(pool: &PgPool, payload: &[u8]) -> Result<()> {
    let message_reader = capnp::serialize::read_message(
        &mut std::io::Cursor::new(payload),
        capnp::message::ReaderOptions::new(),
    )
    .context("Failed to read Cap'n Proto message")?;

    let transcription_reader = message_reader
        .get_root::<transcription_capnp::transcription::Reader>()
        .context("Failed to get transcription root")?;

    let stream_id = transcription_reader
        .get_stream_id()
        .context("Failed to get stream_id")?
        .to_string()
        .context("Failed to convert stream_id to string")?;
    let timestamp_ns = transcription_reader.get_timestamp_ns();
    let text = transcription_reader
        .get_text()
        .context("Failed to get text")?
        .to_string()
        .context("Failed to convert text to string")?;
    let language = transcription_reader
        .get_language()
        .context("Failed to get language")?
        .to_string()
        .context("Failed to convert language to string")?;
    let duration_s = transcription_reader.get_duration_s();
    let segment_start_s = transcription_reader.get_segment_start_s();
    let segment_end_s = transcription_reader.get_segment_end_s();

    let words_reader = transcription_reader
        .get_words()
        .context("Failed to get words")?;

    let mut words = Vec::with_capacity(words_reader.len() as usize);
    for word_reader in words_reader.iter() {
        let word = TranscriptWord {
            word: word_reader
                .get_word()
                .context("Failed to get word")?
                .to_string()
                .context("Failed to convert word to string")?,
            start_s: word_reader.get_start_s(),
            end_s: word_reader.get_end_s(),
            probability: word_reader.get_probability(),
        };
        words.push(word);
    }

    let transcript = Transcript {
        stream_id,
        timestamp_ns,
        text,
        language,
        duration_s,
        segment_start_s,
        segment_end_s,
        words,
    };

    write_transcription_to_postgres(pool, &transcript).await
}
