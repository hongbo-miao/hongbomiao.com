use crate::handlers::get_root::get_root;
use axum::body::to_bytes;
use axum::{
    Router,
    body::Body,
    http::{Request, StatusCode},
};
use tower::ServiceExt;

#[tokio::test]
async fn test_root_handler() {
    let app = Router::new().route("/", axum::routing::get(get_root));
    let res = app
        .oneshot(
            Request::builder()
                .uri("/")
                .method("GET")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    // Use a 1MB limit for the res body
    let body = to_bytes(res.into_body(), 1024 * 1024).await.unwrap();
    let body_str = String::from_utf8(body.to_vec()).unwrap();
    assert_eq!(body_str, "ok");
}
