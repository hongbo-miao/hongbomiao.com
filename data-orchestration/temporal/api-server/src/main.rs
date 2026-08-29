#![deny(dead_code)]
#![deny(unreachable_code)]
#![forbid(unsafe_code)]
#![forbid(unused_must_use)]

pub mod graphql;
pub mod handlers;
pub mod services;
pub mod types;

use std::str::FromStr;

use temporalio_client::{Client, ClientOptions, Connection, ConnectionOptions, Url};

use crate::graphql::schema::create_schema;
use crate::services::serve_api_endpoints::serve_api_endpoints;
use crate::types::application_state::ApplicationState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let temporal_server_url = std::env::var("TEMPORAL_SERVER_URL")?;
    let temporal_namespace =
        std::env::var("TEMPORAL_NAMESPACE").unwrap_or_else(|_| "recording-report".to_string());

    let connection =
        Connection::connect(ConnectionOptions::new(Url::from_str(&temporal_server_url)?).build())
            .await?;
    let temporal_client = Client::new(connection, ClientOptions::new(temporal_namespace).build())?;

    let application_state = ApplicationState {
        schema: create_schema(temporal_client),
    };
    tracing::info!("api-server listening on 0.0.0.0:8085");
    serve_api_endpoints(application_state).await
}
