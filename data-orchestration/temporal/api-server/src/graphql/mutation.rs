use async_graphql::{Context, Object};
use shared::constants::task_queue_name::TASK_QUEUE_NAME;
use shared::types::recording_deletion::RecordingDeletion;
use shared::types::recording_upload::RecordingUpload;
use shared::workflows::delete_recording_workflow::DeleteRecordingWorkflow;
use shared::workflows::recording_report_workflow::RecordingReportWorkflow;
use temporalio_client::{Client, WorkflowStartOptions};
use temporalio_common::protos::temporal::api::enums::v1::WorkflowIdConflictPolicy;

use crate::graphql::types::start_workflow_response::StartWorkflowResponse;

pub struct Mutation;

#[Object]
impl Mutation {
    async fn upload_recording(
        &self,
        ctx: &Context<'_>,
        recording_id: String,
        requester_email: String,
    ) -> Result<StartWorkflowResponse, String> {
        let temporal_client = ctx.data::<Client>().map_err(|error| error.message)?;
        let workflow_id = format!("recording-report-{recording_id}");

        // The workflow ID is derived from the recording ID rather than a random
        // UUID: a retried upload from a flaky connection collides with the
        // existing workflow (UseExisting) instead of starting duplicate,
        // expensive transcription work.
        let start_options = WorkflowStartOptions::new(TASK_QUEUE_NAME, workflow_id.clone())
            .id_conflict_policy(WorkflowIdConflictPolicy::UseExisting)
            .build();

        temporal_client
            .start_workflow(
                RecordingReportWorkflow::run,
                RecordingUpload {
                    recording_id,
                    requester_email,
                },
                start_options,
            )
            .await
            .map_err(|error| format!("Failed to start workflow {workflow_id}: {error}"))?;

        Ok(StartWorkflowResponse { workflow_id })
    }

    async fn delete_recording(
        &self,
        ctx: &Context<'_>,
        recording_id: String,
    ) -> Result<StartWorkflowResponse, String> {
        let temporal_client = ctx.data::<Client>().map_err(|error| error.message)?;
        let workflow_id = format!("recording-deletion-{recording_id}");

        let start_options = WorkflowStartOptions::new(TASK_QUEUE_NAME, workflow_id.clone())
            .id_conflict_policy(WorkflowIdConflictPolicy::UseExisting)
            .build();

        temporal_client
            .start_workflow(
                DeleteRecordingWorkflow::run,
                RecordingDeletion { recording_id },
                start_options,
            )
            .await
            .map_err(|error| format!("Failed to start workflow {workflow_id}: {error}"))?;

        Ok(StartWorkflowResponse { workflow_id })
    }
}
