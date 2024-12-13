use async_graphql::{SimpleObject, Subscription};
use futures_util::stream::Stream;
use std::time::Duration;
use tokio_stream::StreamExt;

#[derive(SimpleObject)]
pub struct CountdownResponse {
    pub count: i32,
}

pub struct Subscription;

#[Subscription]
impl Subscription {
    async fn countdown(&self) -> impl Stream<Item = CountdownResponse> {
        let mut counter = 10;
        tokio_stream::wrappers::IntervalStream::new(tokio::time::interval(Duration::from_secs(1)))
            .map(move |_| {
                counter -= 1;
                CountdownResponse { count: counter }
            })
            .take(10)
    }
}
