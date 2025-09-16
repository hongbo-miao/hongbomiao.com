use crate::handlers::get_root::get_root;
use axum::{Router, routing::get};
use axum_test::TestServer;

#[tokio::test]
async fn test_root_handler() {
    let app = Router::new().route("/", get(get_root));
    let server = TestServer::new(app).expect("Failed to create TestServer");

    let response = server.get("/").await;
    response.assert_status_ok();
    response.assert_text("ok");
}
