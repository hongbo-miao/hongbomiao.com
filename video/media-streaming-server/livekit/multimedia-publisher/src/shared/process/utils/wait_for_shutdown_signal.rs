use tokio::signal::unix::{SignalKind, signal};

pub async fn wait_for_shutdown_signal() {
    let mut terminate =
        signal(SignalKind::terminate()).expect("Failed to register SIGTERM handler");
    let mut interrupt = signal(SignalKind::interrupt()).expect("Failed to register SIGINT handler");

    tokio::select! {
        _ = terminate.recv() => {}
        _ = interrupt.recv() => {}
    }
}
