use crate::shared::speaches::types::transcription_response::TranscriptionResponse;

pub async fn transcribe_audio(
    reqwest_client: &reqwest::Client,
    speaches_base_url: &str,
    audio_data: &[u8],
    model: &str,
) -> Result<TranscriptionResponse, Box<dyn std::error::Error + Send + Sync>> {
    let form = reqwest::multipart::Form::new()
        .part(
            "file",
            reqwest::multipart::Part::bytes(audio_data.to_owned())
                .file_name("audio.flac")
                .mime_str("audio/flac")?,
        )
        .text("model", model.to_string())
        .text("language", "en".to_string())
        .text("vad_filter", "false".to_string())
        .text("word_timestamps", "true".to_string());

    let response = reqwest_client
        .post(format!("{speaches_base_url}/v1/audio/transcriptions"))
        .multipart(form)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unable to read response body".to_string());
        return Err(
            format!("Transcription request failed with status {status}: {error_text}").into(),
        );
    }

    let response_text = response.text().await?;
    let transcription_response: TranscriptionResponse = serde_json::from_str(&response_text)
        .map_err(|error| {
            format!(
                "Failed to parse transcription response: {error}. Response body: {response_text}"
            )
        })?;

    Ok(transcription_response)
}
