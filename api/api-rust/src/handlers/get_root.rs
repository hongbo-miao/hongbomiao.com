/// Health check endpoint
#[utoipa::path(
    get,
    path = "/",
    responses(
        (status = 200, description = "Service is healthy", body = String)
    ),
    tag = "health"
)]
pub async fn get_root() -> &'static str {
    "ok"
}

#[cfg(test)]
#[path = "get_root_test.rs"]
mod tests;
