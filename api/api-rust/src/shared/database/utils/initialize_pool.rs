use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;

pub async fn initialize_pool(
    postgres_url: &str,
    postgres_max_connection_count: u8,
) -> Result<PgPool, sqlx::Error> {
    let pool = PgPoolOptions::new()
        .max_connections(postgres_max_connection_count.into())
        .connect(postgres_url)
        .await?;

    tracing::info!("Database connection pool initialized");

    Ok(pool)
}
