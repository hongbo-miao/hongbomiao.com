use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket};
use futures_util::StreamExt;
use tokio_stream::wrappers::BroadcastStream;

use crate::shared::police_audio_stream::utils::police_audio_stream_manager::{
    PoliceAudioStreamManager, PoliceAudioStreamState,
};

pub async fn handle_police_audio_websocket(
    mut web_socket: WebSocket,
    police_audio_stream_state: Arc<PoliceAudioStreamState>,
    police_stream_id: String,
) {
    let client_id =
        PoliceAudioStreamManager::add_client(&police_audio_stream_state, &police_stream_id).await;
    // subscribe to audio broadcast
    let audio_receiver = {
        match PoliceAudioStreamManager::get_audio_sender(
            &police_audio_stream_state,
            &police_stream_id,
        )
        .await
        {
            Some(sender) => sender.subscribe(),
            None => {
                eprintln!("Stream sender not found for police_stream_id: {police_stream_id}");
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

    PoliceAudioStreamManager::remove_client(
        &police_audio_stream_state,
        &police_stream_id,
        client_id,
    )
    .await;
}
