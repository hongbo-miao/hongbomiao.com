use sqlx::PgPool;

use crate::shared::database::types::pg_graphql_request::PgGraphqlRequest;
use crate::shared::database::utils::resolve_graphql::resolve_graphql;

pub async fn execute_database_graphql(
    pool: &PgPool,
    query: String,
    variables: Option<serde_json::Value>,
    operation_name: Option<String>,
) -> Result<serde_json::Value, String> {
    let request = PgGraphqlRequest {
        query,
        variables,
        operation_name,
    };

    let response = resolve_graphql(pool, request)
        .await
        .map_err(|error| format!("Failed to execute database: {error}"))?;

    match response.errors {
        Some(errors) if !errors.is_empty() => {
            return Err(format!(
                "Database errors: {}",
                serde_json::to_string(&errors).unwrap_or_else(|error| {
                    format!("Could not serialize database errors: {error}")
                })
            ));
        }
        _ => {}
    }

    response
        .data
        .ok_or_else(|| "No data returned from database".to_string())
}
