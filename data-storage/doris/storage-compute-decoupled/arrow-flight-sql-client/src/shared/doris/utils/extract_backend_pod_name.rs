use anyhow::{Result, anyhow};

pub fn extract_backend_pod_name(location_uri: &str) -> Result<String> {
    let (_, host_and_port) = location_uri
        .split_once("://")
        .ok_or_else(|| anyhow!("Location URI has no scheme: {location_uri}"))?;
    let host = host_and_port
        .split(':')
        .next()
        .ok_or_else(|| anyhow!("Location URI has no host: {location_uri}"))?;
    let pod_name = host
        .split('.')
        .next()
        .ok_or_else(|| anyhow!("Location URI host is empty: {location_uri}"))?;
    Ok(pod_name.to_string())
}

#[cfg(test)]
mod tests {
    use super::extract_backend_pod_name;

    #[test]
    fn extracts_pod_name_from_fully_qualified_backend_uri() {
        let location_uri = "grpc+tcp://doris-application-a-0.doris-application-a.storage-compute-decoupled.svc.cluster.local:8050";
        assert_eq!(
            extract_backend_pod_name(location_uri).unwrap(),
            "doris-application-a-0"
        );
    }
}
