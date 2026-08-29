#![deny(dead_code)]
#![deny(unreachable_code)]
#![forbid(unsafe_code)]
#![forbid(unused_must_use)]

use std::str::FromStr;

use shared::activities::recording_activities::RecordingActivities;
use shared::constants::task_queue_name::TASK_QUEUE_NAME;
use shared::workflows::delete_recording_workflow::DeleteRecordingWorkflow;
use shared::workflows::recording_report_workflow::RecordingReportWorkflow;
use temporalio_client::{Client, ClientOptions, Connection, ConnectionOptions, Url};
use temporalio_sdk::{Runtime, Worker, WorkerOptions};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let temporal_server_url = std::env::var("TEMPORAL_SERVER_URL")
        .unwrap_or_else(|_| "http://localhost:7233".to_string());
    let temporal_namespace =
        std::env::var("TEMPORAL_NAMESPACE").unwrap_or_else(|_| "recording-report".to_string());

    let runtime = Runtime::new_assume_tokio(Default::default())?;
    let connection =
        Connection::connect(ConnectionOptions::new(Url::from_str(&temporal_server_url)?).build())
            .await?;
    let client = Client::new(connection, ClientOptions::new(temporal_namespace).build())?;

    // A single worker binary polling one task queue can host any number of
    // workflow types; RecordingReportWorkflow and DeleteRecordingWorkflow are
    // unrelated jobs that happen to share a process and a queue.
    let worker_options = WorkerOptions::new(TASK_QUEUE_NAME)
        .register_workflow::<RecordingReportWorkflow>()?
        .register_workflow::<DeleteRecordingWorkflow>()?
        .register_activities(RecordingActivities)
        .build();

    let mut worker = Worker::new(&runtime, client, worker_options)?;
    tracing::info!("worker polling task queue {TASK_QUEUE_NAME}");
    worker.run().await?;

    Ok(())
}
