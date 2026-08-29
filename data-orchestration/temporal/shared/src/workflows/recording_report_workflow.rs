use std::time::Duration;

use temporalio_common::RetryPolicy;
use temporalio_macros::{workflow, workflow_methods};
use temporalio_sdk::{ActivityOptions, WorkflowContext, WorkflowContextView, WorkflowResult};

use crate::activities::recording_activities::RecordingActivities;
use crate::types::recording_upload::RecordingUpload;
use crate::types::send_report_email_input::SendReportEmailInput;

fn activity_retry_policy() -> RetryPolicy {
    RetryPolicy::builder()
        .initial_interval(Duration::from_secs(1))
        .backoff_coefficient(2.0)
        .maximum_attempts(3)
        .build()
}

#[workflow]
pub struct RecordingReportWorkflow {
    stage: String,
}

impl Default for RecordingReportWorkflow {
    fn default() -> Self {
        Self {
            stage: "started".to_string(),
        }
    }
}

#[workflow_methods]
impl RecordingReportWorkflow {
    #[run]
    pub async fn run(
        ctx: &mut WorkflowContext<Self>,
        recording_upload: RecordingUpload,
    ) -> WorkflowResult<String> {
        // Workflow bodies are replayed from history on every recovery, so they must
        // be deterministic: no direct I/O, no SystemTime::now(), no thread-random,
        // and no tokio::time::sleep here. All of that belongs in activities or
        // ctx.timer(...), which are recorded in history instead of re-executed on
        // replay.
        ctx.state_mut(|state| state.stage = "transcribing".to_string());
        let transcript = ctx
            .execute_activity(
                RecordingActivities::transcribe_recording_audio,
                recording_upload.recording_id,
                ActivityOptions::with_start_to_close_timeout(Duration::from_secs(10))
                    .retry_policy(activity_retry_policy())
                    .build(),
            )
            .await?;

        // A durable timer held by the Temporal server: the window in which the
        // worker can be killed to demonstrate that the workflow survives a crash.
        ctx.state_mut(|state| state.stage = "waiting".to_string());
        ctx.timer(Duration::from_secs(5)).await;

        ctx.state_mut(|state| state.stage = "summarizing".to_string());
        let report = ctx
            .execute_activity(
                RecordingActivities::summarize_recording,
                transcript,
                ActivityOptions::with_start_to_close_timeout(Duration::from_secs(10))
                    .retry_policy(activity_retry_policy())
                    .build(),
            )
            .await?;

        ctx.state_mut(|state| state.stage = "sending_email".to_string());
        ctx.execute_activity(
            RecordingActivities::send_report_email,
            SendReportEmailInput {
                requester_email: recording_upload.requester_email,
                report: report.clone(),
            },
            ActivityOptions::with_start_to_close_timeout(Duration::from_secs(10))
                .retry_policy(activity_retry_policy())
                .build(),
        )
        .await?;

        ctx.state_mut(|state| state.stage = "completed".to_string());
        Ok(report)
    }

    #[query]
    pub fn current_stage(&self, _ctx: &WorkflowContextView) -> String {
        self.stage.clone()
    }
}
