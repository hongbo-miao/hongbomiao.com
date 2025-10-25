use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use tokio::{
    sync::{RwLock, broadcast},
    task::JoinHandle,
};

use crate::shared::fire_audio_stream::utils::process_fire_audio_stream::process_fire_audio_stream;

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
    pub async fn start_stream(state: &FireAudioStreamState, fire_stream_id: &str) {
        // create broadcasters and task if missing
        let mut tasks = state.tasks.write().await;
        if tasks.contains_key(fire_stream_id) {
            return;
        }

        let fire_stream_id_str = fire_stream_id.to_string();
        let (audio_sender, _audio_receiver) = broadcast::channel::<Vec<u8>>(64);

        state
            .audio_senders
            .write()
            .await
            .insert(fire_stream_id_str.clone(), audio_sender.clone());

        let handle = tokio::spawn(process_fire_audio_stream(
            fire_stream_id_str.clone(),
            audio_sender.clone(),
        ));
        tasks.insert(fire_stream_id_str, handle);
    }

    pub async fn add_client(state: &FireAudioStreamState, fire_stream_id: &str) -> u64 {
        let mut next_client_id_lock = state.next_client_id.write().await;
        let id = *next_client_id_lock;
        *next_client_id_lock += 1;
        drop(next_client_id_lock);
        let mut active_client_map = state.active_clients.write().await;
        active_client_map
            .entry(fire_stream_id.to_string())
            .or_default()
            .insert(id);
        id
    }

    pub async fn remove_client(state: &FireAudioStreamState, fire_stream_id: &str, id: u64) {
        let mut active_client_map = state.active_clients.write().await;
        if let Some(set) = active_client_map.get_mut(fire_stream_id) {
            set.remove(&id);
            if set.is_empty() {
                // stop task and clean up
                drop(active_client_map);
                Self::stop_stream(state, fire_stream_id).await;
            }
        }
    }

    pub async fn get_audio_sender(
        state: &FireAudioStreamState,
        fire_stream_id: &str,
    ) -> Option<broadcast::Sender<Vec<u8>>> {
        let guard = state.audio_senders.read().await;
        guard.get(fire_stream_id).cloned()
    }

    async fn stop_stream(state: &FireAudioStreamState, fire_stream_id: &str) {
        if let Some(handle) = state.tasks.write().await.remove(fire_stream_id) {
            handle.abort();
        }
        state.audio_senders.write().await.remove(fire_stream_id);
        state.active_clients.write().await.remove(fire_stream_id);
    }
}
