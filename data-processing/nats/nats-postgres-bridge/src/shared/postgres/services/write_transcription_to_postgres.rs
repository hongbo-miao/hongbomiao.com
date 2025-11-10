use crate::shared::transcripts::types::transcript::Transcript;
use anyhow::Result;
use sqlx::PgPool;
use tracing::{error, info};

pub async fn write_transcription_to_postgres(pool: &PgPool, transcript: &Transcript) -> Result<()> {
    let words_json = serde_json::to_value(&transcript.words)?;

    match sqlx::query(
        r#"
        insert into transcriptions (
            stream_id,
            timestamp_ns,
            text,
            language,
            duration_s,
            segment_start_s,
            segment_end_s,
            words
        )
        values ($1, $2, $3, $4, $5, $6, $7, $8)
        on conflict (stream_id, timestamp_ns) do update
        set
            text = excluded.text,
            language = excluded.language,
            duration_s = excluded.duration_s,
            segment_start_s = excluded.segment_start_s,
            segment_end_s = excluded.segment_end_s,
            words = excluded.words
        "#,
    )
    .bind(&transcript.stream_id)
    .bind(transcript.timestamp_ns)
    .bind(&transcript.text)
    .bind(&transcript.language)
    .bind(transcript.duration_s)
    .bind(transcript.segment_start_s)
    .bind(transcript.segment_end_s)
    .bind(&words_json)
    .execute(pool)
    .await
    {
        Ok(_) => {
            info!(
                "Successfully wrote transcription to Postgres (stream_id={}, timestamp_ns={}): {}",
                transcript.stream_id, transcript.timestamp_ns, transcript.text
            );
            Ok(())
        }
        Err(error) => {
            error!(
                "Failed to write transcription to Postgres (stream_id={}, timestamp_ns={}): {error}",
                transcript.stream_id, transcript.timestamp_ns
            );
            Err(error.into())
        }
    }
}
