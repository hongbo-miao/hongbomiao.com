use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket};
use futures_util::StreamExt;
use tokio_stream::wrappers::BroadcastStream;

use crate::shared::fire_audio_stream::utils::fire_audio_stream_manager::{
    FireAudioStreamManager, FireAudioStreamState,
};

pub async fn handle_fire_audio_websocket(
    mut web_socket: WebSocket,
    fire_audio_stream_state: Arc<FireAudioStreamState>,
    fire_stream_id: String,
) {
    let client_id =
        FireAudioStreamManager::add_client(&fire_audio_stream_state, &fire_stream_id).await;
    // subscribe to audio broadcast
    let audio_receiver = {
        match FireAudioStreamManager::get_audio_sender(&fire_audio_stream_state, &fire_stream_id)
            .await
        {
            Some(sender) => sender.subscribe(),
            None => {
                eprintln!("Stream sender not found for fire_stream_id: {fire_stream_id}");
                let _ = web_socket.send(Message::Close(None)).await;
                return;
            }
        }
    };
    let mut audio_stream = BroadcastStream::new(audio_receiver);

    // forward audio to websocket until disconnect
    loop {
        tokio::select! {
            next = audio_stream.next() => {
                match next {
                    Some(Ok(chunk)) => {
                        if web_socket.send(Message::Binary(chunk.into())).await.is_err() { break; }
                    }
                    Some(Err(_)) => { break; }
                    None => { break; }
                }
            }
            message = web_socket.recv() => {
                // consume client messages to detect disconnect
                match message { Some(Ok(_)) => {}, _ => break }
            }
        }
    }

    FireAudioStreamManager::remove_client(&fire_audio_stream_state, &fire_stream_id, client_id)
        .await;
}
