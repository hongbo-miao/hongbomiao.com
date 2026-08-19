use anyhow::{Result, anyhow};

// Doris returns backend locations as grpc:// or grpc+tcp://, but tonic::transport::Channel only
// accepts http:// / https://. Handing tonic a grpc+tcp:// URI does not fail at parse time; it
// fails deep inside h2 with "Error parsing ':scheme' metadata: error=invalid value", which is why
// this rewrite happens explicitly here rather than relying on tonic to normalize the scheme.
pub fn rewrite_location_uri_scheme(location_uri: &str) -> Result<String> {
    let (_, host_and_port) = location_uri
        .split_once("://")
        .ok_or_else(|| anyhow!("Location URI has no scheme: {location_uri}"))?;
    Ok(format!("http://{host_and_port}"))
}

#[cfg(test)]
mod tests {
    use super::rewrite_location_uri_scheme;

    #[test]
    fn rewrites_grpc_tcp_scheme_to_http() {
        assert_eq!(
            rewrite_location_uri_scheme("grpc+tcp://doris-application-a-0:8050").unwrap(),
            "http://doris-application-a-0:8050"
        );
    }
}
