# Stream AI Inference

A walkthrough of the native real-time AI surface Apache Flink 2.x added to SQL / the Table API, plus Apache Fluss as the live memory layer that turns the job's own output into context for its next prediction:

- **`CREATE MODEL`** — registers a remote AI model in the catalog.
- **`ML_PREDICT()`** — a table-valued function (TVF) for streaming, row-level model inference.
- **`VECTOR_SEARCH()`** — a TVF for inline top-k vector similarity search, for real-time RAG.
- **Fluss lookup joins** — a Flink job cannot read its own output back as a dimension table using state alone. Fluss makes it a lookup join.

```mermaid
flowchart TD
    subgraph streamJob["stream-ai-inference · FlinkDeployment (all SQL)"]
        genTicket["generated_ticket<br/>1 ticket / 5s"]
        genIncident["generated_incident<br/><i>cycles operational → degraded → outage</i>"]

        flussTicket[("fluss: support_ticket (log)")]
        flussIncident[("fluss: service_incident (PK: component)")]
        flussMemory[("fluss: customer_memory (PK: customer_id)")]
        flussInsight[("fluss: ticket_insight (PK: ticket_id)")]

        enrich{{"lookup join × 2<br/><i>customer_memory + service_incident</i>"}}

        triage["ML_PREDICT(triage_model)<br/><i>chat completions → urgency label</i>"]

        embed["ML_PREDICT(embedding_model)<br/><i>embeddings → ARRAY&lt;FLOAT&gt;</i>"]
        search["VECTOR_SEARCH(knowledge_base, …)<br/><i>top-1 nearest article + score</i>"]
        barrier["toDataStream / fromDataStream<br/><i>optimizer barrier — see gotchas</i>"]
        answer["ML_PREDICT(answer_model)<br/><i>answer grounded in article + incident status</i>"]

        join{{"JOIN ON ticket_id<br/><i>state TTL 10 min</i>"}}
        insightSink[/"ticket_insight_sink<br/><i>connector = print → TaskManager stdout</i>"/]
        aggregate["GROUP BY customer_id<br/><i>ticket_count, urgent_count</i>"]

        genTicket --> flussTicket
        genIncident --> flussIncident
        flussTicket --> enrich
        flussMemory -.->|lookup join| enrich
        flussIncident -.->|lookup join| enrich
        enrich --> triage --> join
        enrich --> embed --> search --> barrier --> answer --> join
        join --> insightSink
        join --> flussInsight
        flussInsight --> aggregate --> flussMemory
    end

    subgraph tieringJob["fluss-datalake-tiering · FlinkDeployment"]
        tiering["fluss-flink-tiering jar<br/><i>mirrors datalake-enabled tables<br/>table.datalake.freshness 30s</i>"]
    end

    subgraph reportJob["union-read-report · one-shot Kubernetes Job"]
        report["UnionReadReporter<br/><i>batch mode, run on demand</i>"]
        reportOut[/"banner → stdout<br/><i>$lake vs union counts, snapshot history,<br/>the resolved split</i>"/]
        report --> reportOut
    end

    subgraph externalServices["Outside the jobs"]
        ollama["ollama<br/><i>Service → Ollama on the macOS host, Metal / MPS</i><br/>qwen3:0.6b · all-minilm:22m"]
        knowledgeBaseLoader["knowledge-base-loader<br/><i>one-shot Kubernetes Job</i>"]
        zookeeper["zookeeper<br/><i>required by Fluss 0.9.x</i>"]

        subgraph rustfs["rustfs · S3-compatible object storage"]
            lanceDataset[("vector-store<br/><i>/knowledge-base.lance</i>")]
            paimonWarehouse[("paimon-warehouse<br/><i>/fluss — Paimon lake tier</i>")]
            flussRemoteData[("fluss-remote-data<br/><i>/kv — Fluss KV snapshots</i>")]
            flinkCheckpoints[("flink-checkpoints<br/><i>/stream-ai-inference</i>")]
        end
    end

    triage -.->|/v1/chat/completions| ollama
    answer -.->|/v1/chat/completions| ollama
    embed -.->|/v1/embeddings| ollama
    search -.->|lance connector| lanceDataset
    knowledgeBaseLoader -->|embeds 30 FAQ articles| lanceDataset
    knowledgeBaseLoader -.->|/v1/embeddings| ollama
    flussTicket -.-> zookeeper

    flussInsight -->|table.datalake.enabled| tiering
    flussMemory -->|table.datalake.enabled| tiering
    tiering --> paimonWarehouse
    flussInsight -.->|KV snapshots| flussRemoteData
    flussMemory -.->|KV snapshots| flussRemoteData
    streamJob -.->|checkpoints, Flink's own S3 plugin| flinkCheckpoints

    paimonWarehouse -->|"one table name, two tiers:<br/>paimon — history, from the latest snapshot"| report
    flussInsight -->|"…and the fluss log, read from<br/>the offset that snapshot reached"| report
```
