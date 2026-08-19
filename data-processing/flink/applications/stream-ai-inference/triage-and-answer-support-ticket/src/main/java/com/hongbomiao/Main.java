package com.hongbomiao;

import static com.hongbomiao.fluss.BuildFlussCatalogDdl.buildFlussCatalogDdl;
import static com.hongbomiao.fluss.CreateFlussTableIfAbsent.createFlussTableIfAbsent;

import com.hongbomiao.sources.ServiceIncidentSource;
import com.hongbomiao.sources.SupportTicketSource;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.connector.source.util.ratelimit.RateLimiterStrategy;
import org.apache.flink.connector.datagen.source.DataGeneratorSource;
import org.apache.flink.streaming.api.datastream.DataStreamSource;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.StatementSet;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;

/**
 * Demonstrates the real-time AI building blocks Flink 2 added to SQL / Table API, plus Apache
 * Fluss as the live memory layer that closes the loop between them:
 *
 * <ul>
 *   <li>{@code CREATE MODEL} — registers OpenAI-compatible models (served here by an Ollama
 *       running on the macOS host) in the catalog.
 *   <li>{@code ML_PREDICT()} — streaming row-level inference: ticket triage and embeddings.
 *   <li>{@code VECTOR_SEARCH()} — retrieves the most relevant knowledge-base articles for each
 *       ticket's embedding from a custom {@code lance} connector (see {@code
 *       com.hongbomiao.lance}), closing a small retrieval-augmented-generation loop.
 *   <li><b>Fluss lookup joins</b> — a Flink job cannot read its own output back as a dimension
 *       table using state alone. Fluss makes it a lookup join: {@code customer_memory} is
 *       continuously aggregated from this same job's {@code ticket_insight} output, then
 *       lookup-joined back into the next ticket for that customer. {@code service_incident} is a
 *       second, independently-changing dimension table, so a ticket's prompt reflects live outage
 *       state, not just its own text.
 * </ul>
 */
public class Main {

  private static String getEnv(String name, String defaultValue) {
    String value = System.getenv(name);
    return value == null || value.isEmpty() ? defaultValue : value;
  }

  public static void main(String[] args) throws Exception {
    String ollamaBaseUrl = getEnv("OLLAMA_BASE_URL", "http://ollama:11434");
    String chatModel = getEnv("OLLAMA_CHAT_MODEL", "qwen3:0.6b");
    String embeddingModel = getEnv("OLLAMA_EMBEDDING_MODEL", "all-minilm:22m");
    int embeddingDimension = Integer.parseInt(getEnv("EMBEDDING_DIMENSION", "384"));
    String datasetUri = getEnv("KNOWLEDGE_BASE_DATASET_URI", "s3://vector-store/knowledge-base.lance");
    String s3Endpoint = getEnv("S3_ENDPOINT", "http://rustfs-svc:9000");
    String s3AccessKey = getEnv("S3_ACCESS_KEY", "rustfs_admin");
    String s3SecretKey = getEnv("S3_SECRET_KEY", "passw0rd");
    String flussBootstrapServers =
        getEnv(
            "FLUSS_BOOTSTRAP_SERVERS",
            "coordinator-server-hs.stream-ai-inference.svc.cluster.local:9124");

    StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
    StreamTableEnvironment tableEnv = StreamTableEnvironment.create(env);

    DataGeneratorSource<Row> ticketGeneratorSource =
        new DataGeneratorSource<>(
            new SupportTicketSource(),
            Long.MAX_VALUE,
            RateLimiterStrategy.perSecond(0.2),
            Types.ROW_NAMED(
                new String[] {"ticket_id", "customer_id", "component", "ticket_text"},
                Types.LONG, Types.STRING, Types.STRING, Types.STRING));
    DataStreamSource<Row> ticketStream =
        env.fromSource(
            ticketGeneratorSource, WatermarkStrategy.noWatermarks(), "support-ticket-source");
    tableEnv.createTemporaryView(
        "generated_ticket",
        tableEnv
            .fromDataStream(ticketStream)
            .as("ticket_id", "customer_id", "component", "ticket_text"));

    DataGeneratorSource<Row> incidentGeneratorSource =
        new DataGeneratorSource<>(
            new ServiceIncidentSource(),
            Long.MAX_VALUE,
            RateLimiterStrategy.perSecond(0.1),
            Types.ROW_NAMED(
                new String[] {"component", "incident_status", "incident_summary", "updated_at"},
                Types.STRING, Types.STRING, Types.STRING, Types.LONG));
    DataStreamSource<Row> incidentStream =
        env.fromSource(
            incidentGeneratorSource, WatermarkStrategy.noWatermarks(), "service-incident-source");
    tableEnv.createTemporaryView(
        "generated_incident",
        tableEnv
            .fromDataStream(incidentStream)
            .as("component", "incident_status", "incident_summary", "updated_at"));

    // Fluss tables must live in a Fluss catalog — only FlinkCatalogFactory is SPI-registered, so
    // inline `CREATE TABLE ... WITH ('connector' = 'fluss')` fails with "Could not find any factory
    // for identifier 'fluss'". We never `USE CATALOG`, so Lance and the print sink stay in the
    // default catalog and Fluss tables are always referenced fully qualified.
    //
    // `remote.data.dir`'s own `s3.*` credentials cannot go in this WITH clause — the client gets
    // those from the coordinator (kubernetes/fluss/helm-chart/values.yaml). The `paimon.*` options
    // the helper adds are the opposite case: those must come from here. See the helper's javadoc.
    tableEnv.executeSql(
        buildFlussCatalogDdl(flussBootstrapServers, s3Endpoint, s3AccessKey, s3SecretKey));

    // Fluss tables are durable, so a job restart (upgradeMode stateless, no checkpoint to resume
    // from) must not try to recreate them. SQL's own `IF NOT EXISTS` is not enough once a table is
    // tiered — see the helper's javadoc — so existence is checked before issuing the DDL.
    createFlussTableIfAbsent(
        tableEnv,
        "fluss_catalog",
        "fluss",
        "support_ticket",
        "CREATE TABLE fluss_catalog.fluss.support_ticket (\n"
            + "  ticket_id BIGINT,\n"
            + "  customer_id STRING,\n"
            + "  component STRING,\n"
            + "  ticket_text STRING\n"
            + ") WITH ('bucket.num' = '1')");

    createFlussTableIfAbsent(
        tableEnv,
        "fluss_catalog",
        "fluss",
        "service_incident",
        "CREATE TABLE fluss_catalog.fluss.service_incident (\n"
            + "  component STRING,\n"
            + "  incident_status STRING,\n"
            + "  incident_summary STRING,\n"
            + "  updated_at BIGINT,\n"
            + "  PRIMARY KEY (component) NOT ENFORCED\n"
            + ") WITH ('bucket.num' = '1')");

    // These two are the tables tiered to Paimon (fluss-datalake-tiering.yaml); support_ticket and
    // service_incident are left untiered so the demo shows both a plain Fluss table and a
    // streamhouse one side by side. See README.md's Streamhouse section for '$lake' union reads.
    //
    // The Paimon snapshot retention these two tables need in order to be union-read safely is not
    // set here — Fluss 0.9.1 rejects the per-table 'table.datalake.paimon.' prefix — but on the
    // Paimon catalog instead, as 'datalake.paimon.table-default.snapshot.*' in
    // kubernetes/fluss/helm-chart/values.yaml. That is creation-time only, so it has to be in place
    // before these tables first appear. See README.md's Streamhouse section for why the stock
    // retention is too short for a 30s freshness.
    createFlussTableIfAbsent(
        tableEnv,
        "fluss_catalog",
        "fluss",
        "customer_memory",
        "CREATE TABLE fluss_catalog.fluss.customer_memory (\n"
            + "  customer_id STRING,\n"
            + "  ticket_count BIGINT,\n"
            + "  urgent_count BIGINT,\n"
            + "  PRIMARY KEY (customer_id) NOT ENFORCED\n"
            + ") WITH ('bucket.num' = '1', 'table.datalake.enabled' = 'true',"
            + " 'table.datalake.freshness' = '30s')");

    createFlussTableIfAbsent(
        tableEnv,
        "fluss_catalog",
        "fluss",
        "ticket_insight",
        "CREATE TABLE fluss_catalog.fluss.ticket_insight (\n"
            + "  ticket_id BIGINT,\n"
            + "  customer_id STRING,\n"
            + "  ticket_text STRING,\n"
            + "  urgency STRING,\n"
            + "  matched_article_title STRING,\n"
            + "  match_score DOUBLE,\n"
            + "  suggested_answer STRING,\n"
            + "  PRIMARY KEY (ticket_id) NOT ENFORCED\n"
            + ") WITH ('bucket.num' = '1', 'table.datalake.enabled' = 'true',"
            + " 'table.datalake.freshness' = '30s')");

    tableEnv.executeSql(
        String.format(
            "CREATE MODEL triage_model\n"
                + "INPUT (`input` STRING)\n"
                + "OUTPUT (`content` STRING)\n"
                + "WITH (\n"
                + "  'provider' = 'openai',\n"
                + "  'endpoint' = '%s/v1/chat/completions',\n"
                + "  'api-key' = 'ollama',\n"
                + "  'model' = '%s',\n"
                // qwen3 is a reasoning model: without /no_think it spends most of its tokens on a
                // <think> block, which both slows inference down enormously on CPU and ends up in
                // the output column, where a bare label is wanted.
                + "  'system-prompt' = '/no_think Classify the customer support ticket below as"
                + " exactly one of: urgent, normal, low. Weigh the customer''s prior ticket"
                + " history and the current service status given in the prompt — a routine"
                + " question during an active outage for the affected service, or from a"
                + " customer with a history of urgent tickets, should lean toward urgent. Output"
                + " only the label, nothing else.'\n"
                + ")",
            ollamaBaseUrl, chatModel));

    tableEnv.executeSql(
        String.format(
            "CREATE MODEL embedding_model\n"
                + "INPUT (`input` STRING)\n"
                + "OUTPUT (`embedding` ARRAY<FLOAT>)\n"
                + "WITH (\n"
                + "  'provider' = 'openai',\n"
                + "  'endpoint' = '%s/v1/embeddings',\n"
                + "  'api-key' = 'ollama',\n"
                + "  'model' = '%s',\n"
                + "  'dimension' = '%d'\n"
                + ")",
            ollamaBaseUrl, embeddingModel, embeddingDimension));

    tableEnv.executeSql(
        String.format(
            "CREATE MODEL answer_model\n"
                + "INPUT (`input` STRING)\n"
                + "OUTPUT (`content` STRING)\n"
                + "WITH (\n"
                + "  'provider' = 'openai',\n"
                + "  'endpoint' = '%s/v1/chat/completions',\n"
                + "  'api-key' = 'ollama',\n"
                + "  'model' = '%s',\n"
                + "  'system-prompt' = '/no_think You are a support agent. Using only the"
                + " knowledge-base"
                + " excerpts and the current service status provided, write a two-sentence answer"
                + " to the customer ticket. If the excerpts do not cover the question, say a"
                + " human agent will follow up. The prompt ends with a single opening"
                + " instruction; follow it for the first sentence and never repeat the"
                + " instruction back to the customer.'\n"
                + ")",
            ollamaBaseUrl, chatModel));

    // The columns are named `article_*` rather than after the Lance dataset's own `id` / `title` /
    // `body` / `embedding`: VECTOR_SEARCH puts these columns beside the query table's, and the
    // joined row's field names have to stay unique. The connector matches columns to the dataset by
    // position, so renaming them here is safe.
    tableEnv.executeSql(
        String.format(
            "CREATE TABLE knowledge_base (\n"
                + "  article_id BIGINT,\n"
                + "  article_title STRING,\n"
                + "  article_body STRING,\n"
                + "  article_embedding ARRAY<FLOAT>\n"
                + ") WITH (\n"
                + "  'connector' = 'lance',\n"
                + "  'dataset-uri' = '%s',\n"
                + "  'distance-type' = 'cosine',\n"
                + "  's3.endpoint' = '%s',\n"
                + "  's3.access-key' = '%s',\n"
                + "  's3.secret-key' = '%s',\n"
                + "  's3.path-style-access' = 'true'\n"
                + ")",
            datasetUri, s3Endpoint, s3AccessKey, s3SecretKey));

    tableEnv.executeSql(
        "CREATE TABLE ticket_insight_sink (\n"
            + "  ticket_id BIGINT,\n"
            + "  customer_id STRING,\n"
            + "  ticket_text STRING,\n"
            + "  urgency STRING,\n"
            + "  matched_article_title STRING,\n"
            + "  match_score DOUBLE,\n"
            + "  suggested_answer STRING\n"
            + ") WITH ('connector' = 'print')");

    StatementSet statementSet = tableEnv.createStatementSet();

    statementSet.addInsertSql(
        "INSERT INTO fluss_catalog.fluss.support_ticket SELECT * FROM generated_ticket");
    statementSet.addInsertSql(
        "INSERT INTO fluss_catalog.fluss.service_incident SELECT * FROM generated_incident");

    // Two lookup joins against Fluss primary-key tables, both async by default and requiring the
    // full primary key. `customer_memory` is this same job's own aggregated output (see the final
    // INSERT below) — the loop. `service_incident` is independent world state.
    tableEnv.executeSql(
        "CREATE TEMPORARY VIEW enriched_ticket AS\n"
            + "SELECT t.ticket_id, t.customer_id, t.component, t.ticket_text,\n"
            + "       COALESCE(m.ticket_count, CAST(0 AS BIGINT)) AS prior_ticket_count,\n"
            + "       COALESCE(m.urgent_count, CAST(0 AS BIGINT)) AS prior_urgent_count,\n"
            + "       COALESCE(i.incident_status, 'operational') AS incident_status,\n"
            + "       COALESCE(i.incident_summary, 'no known issues') AS incident_summary\n"
            + "FROM (SELECT *, PROCTIME() AS proc_time FROM fluss_catalog.fluss.support_ticket) AS t\n"
            + "LEFT JOIN fluss_catalog.fluss.customer_memory\n"
            + "  FOR SYSTEM_TIME AS OF t.proc_time AS m\n"
            + "  ON t.customer_id = m.customer_id\n"
            + "LEFT JOIN fluss_catalog.fluss.service_incident\n"
            + "  FOR SYSTEM_TIME AS OF t.proc_time AS i\n"
            + "  ON t.component = i.component");

    // ML_PREDICT is a process table function: it takes the input table as its first argument and
    // returns that table's columns plus the model's output columns. It is *not* a lateral join, so
    // the input table must not also appear on the left of a comma — writing
    // `FROM support_ticket, LATERAL TABLE (ML_PREDICT(TABLE support_ticket, ...))` silently means
    // "cross join every ticket with every prediction", which is both wrong and quadratic.
    tableEnv.executeSql(
        "CREATE TEMPORARY VIEW triage_prompt AS\n"
            + "SELECT ticket_id, customer_id, component, ticket_text, incident_status, incident_summary,\n"
            + "       CONCAT('Ticket: ', ticket_text,\n"
            + "              ' | Customer history: ', CAST(prior_ticket_count AS STRING), ' prior tickets, ',\n"
            + "              CAST(prior_urgent_count AS STRING), ' urgent',\n"
            + "              ' | Service status for ', component, ': ', incident_status,\n"
            + "              ' (', incident_summary, ')') AS prompt\n"
            + "FROM enriched_ticket");

    // The MAP is ML_PREDICT's optional runtime config (MLPredictRuntimeConfigOptions). The default
    // async timeout is far too short for a CPU-only Ollama: the job otherwise dies with
    // "TimeoutException: Async function call has timed out" as soon as the first ticket arrives.
    tableEnv.executeSql(
        "CREATE TEMPORARY VIEW triaged_ticket AS\n"
            + "SELECT ticket_id, customer_id, ticket_text, content AS urgency\n"
            + "FROM TABLE (ML_PREDICT(TABLE triage_prompt, MODEL triage_model,"
            + " DESCRIPTOR(prompt),"
            + " MAP['timeout', '600s', 'max-concurrent-operations', '2']))");

    tableEnv.executeSql(
        "CREATE TEMPORARY VIEW embedded_ticket AS\n"
            + "SELECT ticket_id, ticket_text, incident_status, incident_summary,\n"
            + "       prior_ticket_count, prior_urgent_count, embedding\n"
            + "FROM TABLE (ML_PREDICT(TABLE enriched_ticket, MODEL embedding_model,"
            + " DESCRIPTOR(ticket_text),"
            + " MAP['timeout', '600s', 'max-concurrent-operations', '2']))");

    // VECTOR_SEARCH, unlike ML_PREDICT, *is* a correlated lateral join: the search runs once per
    // left row, keyed by that row's vector. `embedded_ticket.embedding` therefore has to resolve to
    // a correlation reference ($cor0.embedding in the plan). If the left side of the comma is not
    // the table the query vector comes from, the planner produces an uncorrelated cross join
    // instead, and StreamPhysicalVectorSearchTableFunctionRule — which only matches
    // FlinkLogicalCorrelate(_, FlinkLogicalTableFunctionScan) — never fires, failing the job with a
    // bare "not enough rules to produce a node with desired properties: convention=STREAM_PHYSICAL".
    //
    // `SELECT *` is also load-bearing on Flink 2.2.1: *any* projection above this correlate — even
    // one that keeps the query vector — pushes a Calc down through it, which rewrites the correlate
    // into an uncorrelated FlinkLogicalJoin with a dangling field reference and fails to plan the
    // same way. See README.md; hence the DataStream round trip below rather than a narrowing view.
    Table retrievedArticle =
        tableEnv.sqlQuery(
            "SELECT *\n"
                + "FROM embedded_ticket,\n"
                + "LATERAL TABLE (VECTOR_SEARCH(TABLE knowledge_base,"
                + " DESCRIPTOR(article_embedding), embedded_ticket.embedding, 1))");

    // Converting to a DataStream and back splits the job into two independently optimized plans, so
    // the projections and the second ML_PREDICT below cannot be pushed back down into the
    // VECTOR_SEARCH correlate.
    tableEnv.createTemporaryView(
        "retrieved_article", tableEnv.fromDataStream(tableEnv.toDataStream(retrievedArticle)));

    // The CASE resolves to the opening instruction itself, not to a label the model has to branch
    // on: given all three phrasings plus a label, qwen3:0.6b just echoes them all back into the
    // answer. Selecting the branch in SQL keeps only one instruction visible to the model.
    tableEnv.executeSql(
        "CREATE TEMPORARY VIEW answer_prompt AS\n"
            + "SELECT ticket_id, article_title, score,\n"
            + "       CONCAT('Ticket: ', ticket_text, ' | Knowledge base article: ', article_body,\n"
            + "              ' | Current status for the affected service: ', incident_status,\n"
            + "              ' (', incident_summary, ')',\n"
            + "              ' | Opening instruction: ',\n"
            + "              CASE\n"
            + "                WHEN prior_ticket_count = 0\n"
            + "                  THEN 'Open with a normal greeting.'\n"
            + "                WHEN prior_urgent_count * 10 >= prior_ticket_count * 3\n"
            + "                  THEN 'Open by apologising for the repeated trouble.'\n"
            + "                ELSE 'Open by briefly acknowledging that they have contacted us"
            + " before.'\n"
            + "              END) AS prompt\n"
            + "FROM retrieved_article");

    tableEnv.executeSql(
        "CREATE TEMPORARY VIEW answered_ticket AS\n"
            + "SELECT ticket_id, article_title, score, content AS suggested_answer\n"
            + "FROM TABLE (ML_PREDICT(TABLE answer_prompt, MODEL answer_model,"
            + " DESCRIPTOR(prompt),"
            + " MAP['timeout', '600s', 'max-concurrent-operations', '2']))");

    // Regular (unbounded) join, not an interval join — the source has no time attribute to window
    // on. Cap the state with a TTL so both sides don't buffer forever.
    tableEnv.getConfig().set("table.exec.state.ttl", "10 min");

    tableEnv.executeSql(
        "CREATE TEMPORARY VIEW merged_insight AS\n"
            + "SELECT t.ticket_id, t.customer_id, t.ticket_text, t.urgency,\n"
            + "       a.article_title, a.score, a.suggested_answer\n"
            + "FROM triaged_ticket AS t\n"
            + "JOIN answered_ticket AS a ON t.ticket_id = a.ticket_id");

    statementSet.addInsertSql("INSERT INTO ticket_insight_sink SELECT * FROM merged_insight");
    statementSet.addInsertSql(
        "INSERT INTO fluss_catalog.fluss.ticket_insight SELECT * FROM merged_insight");

    // Closes the loop: aggregates the model's own output back into customer_memory. Reading a Fluss
    // PK table as a source yields a changelog, so this runs in retract mode — stick to COUNT/SUM,
    // which are retract-safe; LAST_VALUE is not reliably supported over a retracting input.
    statementSet.addInsertSql(
        "INSERT INTO fluss_catalog.fluss.customer_memory\n"
            + "SELECT customer_id,\n"
            + "       COUNT(*) AS ticket_count,\n"
            + "       SUM(CASE WHEN urgency = 'urgent' THEN 1 ELSE 0 END) AS urgent_count\n"
            + "FROM fluss_catalog.fluss.ticket_insight\n"
            + "GROUP BY customer_id");

    statementSet.execute();
  }
}
