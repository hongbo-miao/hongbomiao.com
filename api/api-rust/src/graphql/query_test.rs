use crate::graphql::query::Query;
use async_graphql::{EmptyMutation, EmptySubscription, Schema};

#[tokio::test]
async fn test_hello_query() {
    let schema = Schema::new(Query, EmptyMutation, EmptySubscription);
    let query = "
        query {
            hello {
                message
            }
        }
    ";
    let res = schema.execute(query).await;
    let json = serde_json::to_value(&res).unwrap();
    assert!(res.is_ok());
    assert_eq!(
        json["data"]["hello"]["message"].as_str().unwrap(),
        "Hello, world!"
    );
}
