# Storage-Compute Decoupled

```mermaid
flowchart TD
    subgraph rustfs["rustfs · S3-compatible object storage"]
        vault[("doris-storage-vault<br/><i>Doris internal tables</i>")]
        paimonWarehouse[("paimon-warehouse<br/><i>sensor_db.sensor_reading</i>")]
        flinkCheckpoints[("flink-checkpoints")]
    end

    subgraph seed["paimon-seed · one-shot Flink session cluster"]
        datagen["datagen source<br/><i>2M rows · 500 devices</i>"]
        seedJob["paimon-seed-submit<br/><i>sql-client.sh --file seed.sql</i>"]
        datagen --> seedJob
    end

    subgraph fdb["FoundationDB · double redundancy"]
        fdbCluster[("log ×2, storage ×2")]
    end

    subgraph ranger["ranger · authorization"]
        rangerAdmin["ranger-admin<br/><i>policies + service def</i>"]
        rangerDb[("ranger-db · Postgres")]
        rangerSolr[("ranger-solr · audit log")]
        rangerAdmin --> rangerDb
        rangerAdmin -.->|"audit store (stays empty, see Audit)"| rangerSolr
    end

    subgraph doris["doris · DorisDisaggregatedCluster"]
        ms["Meta Service ×2"]
        fe["FE ×1"]
        applicationACG["compute group: application_a<br/><i>SELECT + row filter + column mask</i>"]
        applicationBCG["compute group: application_b<br/><i>SELECT, unfiltered</i>"]
        ms <-->|transactions, metadata| fdbCluster
        fe --> ms
    end

    client["arrow-flight-sql-client<br/><i>Rust, in-cluster</i>"]

    seedJob --> paimonWarehouse
    seedJob -.->|checkpoints| flinkCheckpoints
    ms -->|default storage vault| vault
    fe -->|"pull policies every 30s<br/>access_controller_type=ranger-doris"| rangerAdmin
    client -->|"http://doris-fe-0.doris-fe-internal:8070<br/>handshake + execute() — control path only"| fe
    client -->|"application_a_user<br/>do_get() direct to BE"| applicationACG
    client -->|"application_b_user<br/>do_get() direct to BE"| applicationBCG
    applicationACG --> paimonWarehouse
    applicationBCG --> paimonWarehouse
```

## Develop

```bash
just storage-vault-up
just paimon-catalog-up
just ranger-policies-up
just users-up
just same-table-both-groups-application-a
just same-table-both-groups-application-b
just row-filter-and-masking-application-a
just row-filter-and-masking-application-b
```

`ranger-policies-up` has to come before `users-up`, not after it. `users-up` ends with `set property ... default_compute_group`, and Doris checks `USAGE` on that compute group while setting it -- a check Ranger now answers, so without the compute group policies in place it fails with `CURRENT_USER_NO_AUTH_TO_USE_COMPUTE_GROUP`. The policies do not need the Doris users to exist first: Ranger keeps its own user list, populated by the bootstrap Job.

Ranger Admin UI: <http://localhost:6080> (`admin` / `Passw0rd1`).
