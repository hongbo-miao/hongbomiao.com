use std::time::Duration;

use tracing::{Instrument, info, info_span};
use wtransport::endpoint::endpoint_side::Server;
use wtransport::{Endpoint, Identity, ServerConfig};

use crate::config::AppConfig;
use crate::webtransport::utils::serve_incoming_session::serve_incoming_session;

pub struct WebTransportServer {
    endpoint: Endpoint<Server>,
}

impl WebTransportServer {
    pub async fn create(port_number: u16) -> anyhow::Result<Self> {
        let config = AppConfig::get();
        let identity = Identity::load_pemfiles(
            &config.webtransport_certificate_path,
            &config.webtransport_private_key_path,
        )
        .await?;

        let config = ServerConfig::builder()
            .with_bind_default(port_number)
            .with_identity(identity)
            .keep_alive_interval(Some(Duration::from_secs(3)))
            .build();

        let endpoint = Endpoint::server(config)?;

        Ok(Self { endpoint })
    }

    pub fn get_local_port(&self) -> u16 {
        self.endpoint
            .local_addr()
            .expect("Failed to get local address")
            .port()
    }

    pub async fn serve(self) -> anyhow::Result<()> {
        info!(
            "{}",
            format!(
                "WebTransport server running on port {}",
                self.get_local_port()
            )
        );

        for connection_number in 0.. {
            let incoming_session = self.endpoint.accept().await;

            tokio::spawn(
                serve_incoming_session(incoming_session)
                    .instrument(info_span!("WebTransportConnection", connection_number)),
            );
        }

        Ok(())
    }
}
