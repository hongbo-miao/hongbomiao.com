use crate::shared::database::utils::execute_database_graphql::execute_database_graphql;
use crate::shared::openai::types::chat_response::ChatResponse;
use crate::shared::openai::utils::chat::chat;
use crate::shared::parallel_calculation::types::calculation_response::CalculationResponse;
use crate::shared::parallel_calculation::utils::calculate_parallel::calculate_parallel;
use crate::shared::police_audio_stream::constants::police_streams::POLICE_STREAMS;
use crate::shared::police_audio_stream::utils::police_stream_state::POLICE_STREAM_STATE;
use crate::shared::python_parallel_calculation::types::python_calculation_response::PythonCalculationResponse;
use crate::shared::python_parallel_calculation::utils::calculate_with_python::calculate_with_python;
use async_graphql::{Context, Object, SimpleObject};
use sqlx::PgPool;
use utoipa::ToSchema;

#[derive(SimpleObject, ToSchema)]
pub struct HelloResponse {
    pub message: String,
}

#[derive(SimpleObject, ToSchema)]
pub struct PoliceStream {
    pub id: String,
    pub name: String,
    pub location: String,
}

#[derive(SimpleObject, ToSchema)]
pub struct PoliceStreamStatus {
    pub id: String,
    pub client_count: i32,
}

pub struct Query;

#[Object]
impl Query {
    async fn hello(&self) -> HelloResponse {
        HelloResponse {
            message: "Hello, world!".to_string(),
        }
    }

    async fn chat(&self, message: String) -> Result<ChatResponse, String> {
        chat(message).await
    }

    #[graphql(name = "parallelCalculation")]
    async fn parallel_calculation(&self, items_count: i32) -> Result<CalculationResponse, String> {
        if items_count <= 0 {
            return Err("items_count must be greater than 0".to_string());
        }
        if items_count > 100_000_000 {
            return Err("items_count must be less than or equal to 100,000,000".to_string());
        }

        // Run the blocking Rayon computation in a blocking task
        let result = tokio::task::spawn_blocking(move || calculate_parallel(items_count as usize))
            .await
            .map_err(|error| format!("Task join error: {error}"))?;

        Ok(result)
    }

    #[graphql(name = "pythonParallelCalculation")]
    async fn python_parallel_calculation(
        &self,
        items_count: i32,
    ) -> Result<PythonCalculationResponse, String> {
        if items_count <= 0 {
            return Err("items_count must be greater than 0".to_string());
        }
        if items_count > 100_000_000 {
            return Err("items_count must be less than or equal to 100,000,000".to_string());
        }

        // Run the blocking Python computation in a blocking task
        let result = tokio::task::spawn_blocking(move || calculate_with_python(items_count))
            .await
            .map_err(|error| format!("Task join error: {error}"))?
            .map_err(|error| format!("Python calculation error: {error}"))?;

        Ok(result)
    }

    #[graphql(name = "policeStreams")]
    async fn police_streams(&self) -> Vec<PoliceStream> {
        POLICE_STREAMS
            .iter()
            .map(|(id, info)| PoliceStream {
                id: (*id).to_string(),
                name: info.name.to_string(),
                location: info.location.to_string(),
            })
            .collect()
    }

    #[graphql(name = "policeStreamStatus")]
    async fn police_stream_status(&self) -> Vec<PoliceStreamStatus> {
        let active = POLICE_STREAM_STATE.active_clients.read().await;
        active
            .iter()
            .map(|(k, v)| PoliceStreamStatus {
                id: k.clone(),
                client_count: v.len() as i32,
            })
            .collect()
    }

    async fn database(
        &self,
        ctx: &Context<'_>,
        query: String,
        variables: Option<serde_json::Value>,
        operation_name: Option<String>,
    ) -> Result<serde_json::Value, String> {
        let pool = ctx
            .data::<PgPool>()
            .map_err(|error| format!("Failed to get database pool: {error:?}"))?;

        execute_database_graphql(pool, query, variables, operation_name).await
    }
}

#[cfg(test)]
#[path = "query_test.rs"]
mod tests;
