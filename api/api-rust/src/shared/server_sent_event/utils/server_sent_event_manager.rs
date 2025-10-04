use once_cell::sync::Lazy;
use tokio::sync::broadcast;

use crate::shared::server_sent_event::types::server_sent_event_message::ServerSentEventMessage;

pub static SERVER_SENT_EVENT_SENDER: Lazy<broadcast::Sender<String>> = Lazy::new(|| {
    let (sender, _) = broadcast::channel(1000);
    sender
});

pub struct ServerSentEventManager;

impl ServerSentEventManager {
    pub fn new() -> Self {
        Self
    }

    pub fn subscribe(&self) -> broadcast::Receiver<String> {
        SERVER_SENT_EVENT_SENDER.subscribe()
    }

    pub async fn publish_json(
        &self,
        payload: &ServerSentEventMessage,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let json_data = serde_json::to_string(payload)?;
        let line = format!("data: {json_data}\n\n");

        match SERVER_SENT_EVENT_SENDER.send(line) {
            Ok(_) => Ok(()),
            Err(broadcast::error::SendError(_)) => {
                // No active subscribers, which is fine
                Ok(())
            }
        }
    }
}

pub static SERVER_SENT_EVENT_MANAGER: Lazy<ServerSentEventManager> =
    Lazy::new(ServerSentEventManager::new);
