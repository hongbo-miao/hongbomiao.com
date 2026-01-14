use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tokio::sync::{RwLock, broadcast};
use tokio::task::JoinHandle;

use crate::shared::police_audio_stream::constants::police_streams::POLICE_STREAMS;
use crate::shared::police_audio_stream::utils::process_police_audio_stream::process_police_audio_stream;

#[derive(Clone)]
pub struct PoliceAudioStreamState {
    // active websocket client counts per stream
    pub active_clients: Arc<RwLock<HashMap<String, HashSet<u64>>>>,
    // audio broadcaster per stream (binary PCM chunks)
    pub audio_senders: Arc<RwLock<HashMap<String, broadcast::Sender<Vec<u8>>>>>,
    // server-sent event broadcaster per stream (transcription events)
    pub server_sent_event_senders: Arc<RwLock<HashMap<String, broadcast::Sender<String>>>>,
    // decoder task handles per stream
    pub tasks: Arc<RwLock<HashMap<String, JoinHandle<()>>>>,
    // id counter for WebSocket connection bookkeeping
    pub next_client_id: Arc<RwLock<u64>>,
}

impl PoliceAudioStreamState {
    pub fn new() -> Self {
        Self {
            active_clients: Arc::new(RwLock::new(HashMap::new())),
            audio_senders: Arc::new(RwLock::new(HashMap::new())),
            server_sent_event_senders: Arc::new(RwLock::new(HashMap::new())),
            tasks: Arc::new(RwLock::new(HashMap::new())),
            next_client_id: Arc::new(RwLock::new(1)),
        }
    }
}

pub struct PoliceAudioStreamManager;

impl PoliceAudioStreamManager {
    pub async fn start_stream(state: &PoliceAudioStreamState, police_stream_id: &str) {
        // create broadcasters and task if missing
        let mut tasks = state.tasks.write().await;
        if tasks.contains_key(police_stream_id) {
            return;
        }

        let police_stream_id_str = police_stream_id.to_string();
        let (audio_sender, _audio_receiver) = broadcast::channel::<Vec<u8>>(64);
        let (server_sent_event_sender, _server_sent_event_receiver) =
            broadcast::channel::<String>(64);

        state
            .audio_senders
            .write()
            .await
            .insert(police_stream_id_str.clone(), audio_sender.clone());

        state.server_sent_event_senders.write().await.insert(
            police_stream_id_str.clone(),
            server_sent_event_sender.clone(),
        );

        let stream_url = POLICE_STREAMS
            .get(police_stream_id)
            .map(|stream_info| stream_info.stream_url)
            .expect("Stream ID should exist in POLICE_STREAMS")
            .to_string();
        let handle = tokio::spawn(process_police_audio_stream(
            police_stream_id_str.clone(),
            stream_url.clone(),
            audio_sender.clone(),
            server_sent_event_sender.clone(),
        ));
        tasks.insert(police_stream_id_str, handle);
    }

    pub async fn add_client(state: &PoliceAudioStreamState, police_stream_id: &str) -> u64 {
        let mut next_client_id_lock = state.next_client_id.write().await;
        let id = *next_client_id_lock;
        *next_client_id_lock += 1;
        drop(next_client_id_lock);
        let mut active_client_map = state.active_clients.write().await;
        active_client_map
            .entry(police_stream_id.to_string())
            .or_default()
            .insert(id);
        id
    }

    pub async fn remove_client(state: &PoliceAudioStreamState, police_stream_id: &str, id: u64) {
        let mut active_client_map = state.active_clients.write().await;
        if let Some(set) = active_client_map.get_mut(police_stream_id) {
            set.remove(&id);
            if set.is_empty() {
                // stop task and clean up
                drop(active_client_map);
                Self::stop_stream(state, police_stream_id).await;
            }
        }
    }

    pub async fn get_audio_sender(
        state: &PoliceAudioStreamState,
        police_stream_id: &str,
    ) -> Option<broadcast::Sender<Vec<u8>>> {
        let guard = state.audio_senders.read().await;
        guard.get(police_stream_id).cloned()
    }

    async fn stop_stream(state: &PoliceAudioStreamState, police_stream_id: &str) {
        if let Some(handle) = state.tasks.write().await.remove(police_stream_id) {
            handle.abort();
        }
        state.audio_senders.write().await.remove(police_stream_id);
        state
            .server_sent_event_senders
            .write()
            .await
            .remove(police_stream_id);
        state.active_clients.write().await.remove(police_stream_id);
    }
}
