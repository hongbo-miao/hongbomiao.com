use temporalio_macros::activities;
use temporalio_sdk::activities::{ActivityContext, ActivityError};

use crate::types::send_report_email_input::SendReportEmailInput;

pub struct RecordingActivities;

// The SDK's #[activities] macro requires every activity to live in one impl
// block, which is why all three are defined here rather than one per file.
#[activities]
impl RecordingActivities {
    #[activity]
    pub async fn transcribe_recording_audio(
        ctx: ActivityContext,
        recording_id: String,
    ) -> Result<String, ActivityError> {
        // A real implementation would call a vLLM-hosted speech-to-text model here.
        // Fails on its first two attempts so the workflow's retry policy is visible
        // in the worker log and in the Web UI's pending-activity panel.
        if ctx.info().attempt < 3 {
            return Err(ActivityError::from(anyhow::anyhow!(
                "Simulated transient failure transcribing recording {recording_id}"
            )));
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        let transcript = format!(
            "Transcript for recording {recording_id}: introduction, discussion, action items."
        );
        Ok(transcript)
    }

    #[activity]
    pub async fn summarize_recording(
        _ctx: ActivityContext,
        transcript: String,
    ) -> Result<String, ActivityError> {
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        Ok(format!("Recording summary: {transcript}"))
    }

    #[activity]
    pub async fn send_report_email(
        _ctx: ActivityContext,
        input: SendReportEmailInput,
    ) -> Result<(), ActivityError> {
        // A real implementation must be idempotent: Temporal guarantees activities
        // run at-least-once, so a retried attempt could otherwise send a duplicate email.
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        tracing::info!("Sent report to {}: {}", input.requester_email, input.report);
        Ok(())
    }

    #[activity]
    pub async fn delete_recording_files(
        _ctx: ActivityContext,
        recording_id: String,
    ) -> Result<(), ActivityError> {
        // A real implementation would delete the audio file and its transcript
        // from object storage; this only simulates the I/O.
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        tracing::info!("Deleted recording {recording_id}");
        Ok(())
    }
}
