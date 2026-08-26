# Multi-Cloud Data Pipeline

```mermaid
flowchart TD
    edge["edge device"] --> lb["edge load balancer\n(HAProxy)"]
    lb --> gwWest["sensor-gateway\n(west)"]
    lb --> gwEast["sensor-gateway\n(east)"]

    gwWest --> pulsarWest["Pulsar\n(west)"]
    gwEast --> pulsarEast["Pulsar\n(east)"]
    pulsarWest <-->|geo-replication| pulsarEast

    pulsarWest --> flinkWest["Flink\n(west)"]
    pulsarEast --> flinkEast["Flink\n(east)"]

    flinkWest --> paimonWest["Paimon\n@ west object storage"]
    flinkEast --> paimonEast["Paimon\n@ east object storage"]

    paimonWest --> dorisWest["Doris\n(west)"]
    paimonEast --> dorisEast["Doris\n(east)"]

    dorisWest --> client["clients read local Doris"]
    dorisEast --> client
```

## Usage

```bash
just kind-up
just install-cilium
just connect-cluster-mesh
just create-namespace

just install-pulsar
just initialize-geo-replication
just test-geo-replication

just deploy-gateway
just edge-up
just edge-device-up
just test-ingest

just install-rustfs
just install-flink
just submit-telemetry-sink
just paimon-count

just install-doris
just storage-vault-up
just paimon-catalog-up
just deploy-doris-fe-global-service
just doris-count
```

Tear down with:

```bash
just edge-device-down
just edge-down
just kind-down
```
