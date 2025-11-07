use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;

pub async fn initialize_pool(
    database_url: &str,
    max_connection_count: u8,
) -> Result<PgPool, sqlx::Error> {
    let pool = PgPoolOptions::new()
        .max_connections(max_connection_count.into())
        .connect(database_url)
        .await?;

    tracing::info!("Database connection pool initialized");

    Ok(pool)
}
