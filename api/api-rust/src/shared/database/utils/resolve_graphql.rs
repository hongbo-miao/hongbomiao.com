use crate::shared::database::types::pg_graphql_request::PgGraphqlRequest;
use crate::shared::database::types::pg_graphql_response::PgGraphqlResponse;
use sqlx::PgPool;

pub async fn resolve_graphql(
    pool: &PgPool,
    request: PgGraphqlRequest,
) -> Result<PgGraphqlResponse, sqlx::Error> {
    let variables_json = request.variables.unwrap_or_else(|| serde_json::json!({}));
    let operation_name = request.operation_name.as_deref();

    let result: (serde_json::Value,) = if let Some(operation_name_value) = operation_name {
        sqlx::query_as(
            "select graphql.resolve($1::text, variables => $2::jsonb, operation_name => $3::text)",
        )
        .bind(&request.query)
        .bind(&variables_json)
        .bind(operation_name_value)
        .fetch_one(pool)
        .await?
    } else {
        sqlx::query_as("select graphql.resolve($1::text, variables => $2::jsonb)")
            .bind(&request.query)
            .bind(&variables_json)
            .fetch_one(pool)
            .await?
    };

    let json_result = result.0;

    let response = PgGraphqlResponse {
        data: json_result.get("data").cloned(),
        errors: json_result.get("errors").cloned().and_then(|errors| {
            if errors.is_null() {
                None
            } else {
                Some(errors.as_array().unwrap_or(&vec![]).clone())
            }
        }),
    };

    Ok(response)
}
