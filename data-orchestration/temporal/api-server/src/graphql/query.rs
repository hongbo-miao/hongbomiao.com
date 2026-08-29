use async_graphql::{Context, Object};
use shared::workflows::recording_report_workflow::RecordingReportWorkflow;
use temporalio_client::{Client, WorkflowQueryOptions};

pub struct Query;

#[Object]
impl Query {
    async fn recording_stage(
        &self,
        ctx: &Context<'_>,
        recording_id: String,
    ) -> Result<String, String> {
        let temporal_client = ctx.data::<Client>().map_err(|error| error.message)?;
        let workflow_id = format!("recording-report-{recording_id}");

        temporal_client
            .get_workflow_handle::<RecordingReportWorkflow>(workflow_id)
            .query(
                RecordingReportWorkflow::current_stage,
                (),
                WorkflowQueryOptions::default(),
            )
            .await
            .map_err(|error| format!("Failed to query recording {recording_id}: {error}"))
    }
}
