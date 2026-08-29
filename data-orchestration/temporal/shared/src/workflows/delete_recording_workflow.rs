use std::time::Duration;

use temporalio_macros::{workflow, workflow_methods};
use temporalio_sdk::{ActivityOptions, WorkflowContext, WorkflowResult};

use crate::activities::recording_activities::RecordingActivities;
use crate::types::recording_deletion::RecordingDeletion;

#[workflow]
pub struct DeleteRecordingWorkflow;

impl Default for DeleteRecordingWorkflow {
    fn default() -> Self {
        Self
    }
}

#[workflow_methods]
impl DeleteRecordingWorkflow {
    #[run]
    pub async fn run(
        ctx: &mut WorkflowContext<Self>,
        recording_deletion: RecordingDeletion,
    ) -> WorkflowResult<()> {
        ctx.execute_activity(
            RecordingActivities::delete_recording_files,
            recording_deletion.recording_id,
            ActivityOptions::with_start_to_close_timeout(Duration::from_secs(10)).build(),
        )
        .await?;
        Ok(())
    }
}
