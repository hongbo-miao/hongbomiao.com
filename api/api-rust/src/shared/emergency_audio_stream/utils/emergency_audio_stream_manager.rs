use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use tokio::{
    sync::{RwLock, broadcast},
    task::JoinHandle,
};

use crate::shared::emergency_audio_stream::utils::process_emergency_audio_stream::process_emergency_audio_stream;

#[derive(Clone)]
pub struct FireAudioStreamState {
    // active websocket client counts per stream
    pub active_clients: Arc<RwLock<HashMap<String, HashSet<u64>>>>,
    // audio broadcaster per stream (binary audio chunks)
    pub audio_senders: Arc<RwLock<HashMap<String, broadcast::Sender<Vec<u8>>>>>,
    // consumer task handles per stream
    pub tasks: Arc<RwLock<HashMap<String, JoinHandle<()>>>>,
    // id counter for WebSocket connection bookkeeping
    pub next_client_id: Arc<RwLock<u64>>,
}

impl FireAudioStreamState {
    pub fn new() -> Self {
        Self {
            active_clients: Arc::new(RwLock::new(HashMap::new())),
            audio_senders: Arc::new(RwLock::new(HashMap::new())),
            tasks: Arc::new(RwLock::new(HashMap::new())),
            next_client_id: Arc::new(RwLock::new(1)),
        }
    }
}

pub struct FireAudioStreamManager;

impl FireAudioStreamManager {
    pub async fn start_stream(state: &FireAudioStreamState, emergency_stream_id: &str) {
        // create broadcasters and task if missing
        let mut tasks = state.tasks.write().await;
        if tasks.contains_key(emergency_stream_id) {
            return;
        }

        let emergency_stream_id_str = emergency_stream_id.to_string();
        let (audio_sender, _audio_receiver) = broadcast::channel::<Vec<u8>>(64);

        state
            .audio_senders
            .write()
            .await
            .insert(emergency_stream_id_str.clone(), audio_sender.clone());

        let handle = tokio::spawn(process_emergency_audio_stream(
            emergency_stream_id_str.clone(),
            audio_sender.clone(),
        ));
        tasks.insert(emergency_stream_id_str, handle);
    }

    pub async fn add_client(state: &FireAudioStreamState, emergency_stream_id: &str) -> u64 {
        let mut next_client_id_lock = state.next_client_id.write().await;
        let id = *next_client_id_lock;
        *next_client_id_lock += 1;
        drop(next_client_id_lock);
        let mut active_client_map = state.active_clients.write().await;
        active_client_map
            .entry(emergency_stream_id.to_string())
            .or_default()
            .insert(id);
        id
    }

    pub async fn remove_client(state: &FireAudioStreamState, emergency_stream_id: &str, id: u64) {
        let mut active_client_map = state.active_clients.write().await;
        if let Some(set) = active_client_map.get_mut(emergency_stream_id) {
            set.remove(&id);
            if set.is_empty() {
                // stop task and clean up
                drop(active_client_map);
                Self::stop_stream(state, emergency_stream_id).await;
            }
        }
    }

    pub async fn get_audio_sender(
        state: &FireAudioStreamState,
        emergency_stream_id: &str,
    ) -> Option<broadcast::Sender<Vec<u8>>> {
        let guard = state.audio_senders.read().await;
        guard.get(emergency_stream_id).cloned()
    }

    async fn stop_stream(state: &FireAudioStreamState, emergency_stream_id: &str) {
        if let Some(handle) = state.tasks.write().await.remove(emergency_stream_id) {
            handle.abort();
        }
        state
            .audio_senders
            .write()
            .await
            .remove(emergency_stream_id);
        state
            .active_clients
            .write()
            .await
            .remove(emergency_stream_id);
    }
}
