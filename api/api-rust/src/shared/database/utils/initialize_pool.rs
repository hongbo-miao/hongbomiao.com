use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;

pub async fn initialize_pool(database_url: &str) -> Result<PgPool, sqlx::Error> {
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(database_url)
        .await?;

    tracing::info!("Database connection pool initialized");

    Ok(pool)
}
