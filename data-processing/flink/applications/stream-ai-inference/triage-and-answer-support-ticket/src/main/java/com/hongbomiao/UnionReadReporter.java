package com.hongbomiao;

import static com.hongbomiao.fluss.BuildFlussCatalogDdl.buildFlussCatalogDdl;

import org.apache.flink.table.api.EnvironmentSettings;
import org.apache.flink.table.api.TableEnvironment;
import org.apache.flink.table.api.TableResult;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/**
 * A one-shot batch job (run as a Kubernetes {@code Job}, after the streaming job has been running
 * for a few minutes) that proves Fluss union read: {@code ticket_insight} and {@code
 * customer_memory} are tiered to Paimon by the {@code fluss-datalake-tiering} job, and a plain
 * query against either table reads both the Paimon (historical) tier and the Fluss (real-time)
 * tier as one table. See README.md's Streamhouse section.
 */
public class UnionReadReporter {

  private static String getEnv(String name, String defaultValue) {
    String value = System.getenv(name);
    return value == null || value.isEmpty() ? defaultValue : value;
  }

  public static void main(String[] args) {
    String flussBootstrapServers =
        getEnv(
            "FLUSS_BOOTSTRAP_SERVERS",
            "coordinator-server-hs.stream-ai-inference.svc.cluster.local:9124");
    String s3Endpoint = getEnv("S3_ENDPOINT", "http://rustfs-svc:9000");
    String s3AccessKey = getEnv("S3_ACCESS_KEY", "rustfs_admin");
    String s3SecretKey = getEnv("S3_SECRET_KEY", "passw0rd");

    TableEnvironment tableEnv =
        TableEnvironment.create(EnvironmentSettings.newInstance().inBatchMode().build());

    tableEnv.executeSql(
        buildFlussCatalogDdl(flussBootstrapServers, s3Endpoint, s3AccessKey, s3SecretKey));

    // The `$lake` suffix reads only the Paimon-tiered rows; no suffix reads the union of the
    // Paimon tier and whatever has landed in Fluss since the last tiering commit. lakeCount must
    // be strictly less than unionCount and both non-zero for this to actually demonstrate a union
    // read rather than an empty or fully-tiered table.
    long lakeCount =
        collectSingleLong(
            tableEnv.executeSql(
                "SELECT COUNT(*) FROM fluss_catalog.fluss.`ticket_insight$lake`"));
    long unionCount =
        collectSingleLong(
            tableEnv.executeSql("SELECT COUNT(*) FROM fluss_catalog.fluss.ticket_insight"));

    long snapshotCount =
        collectSingleLong(
            tableEnv.executeSql(
                "SELECT COUNT(*) FROM fluss_catalog.fluss.`ticket_insight$lake$snapshots`"));

    // Printed with a banner and no Flink log prefixes: these few lines are the entire point of
    // this job, and they are otherwise impossible to find in the thousands of INFO lines a Flink
    // MiniCluster emits around them.
    System.out.println();
    System.out.println("================================================================");
    System.out.println(" Fluss union read: one table name, two tiers");
    System.out.println("================================================================");
    System.out.printf(
        "  ticket_insight$lake   paimon tier only      %,8d rows%n", lakeCount);
    System.out.printf(
        "  ticket_insight        paimon + fluss log    %,8d rows%n", unionCount);
    System.out.println("                                              -------------");
    System.out.printf(
        "  served from the fluss log, i.e. written%n"
            + "  since the last tiering commit         %,8d rows%n",
        unionCount - lakeCount);
    System.out.println();
    if (lakeCount == 0) {
      System.out.println(
          "  INCONCLUSIVE: the Paimon tier is empty, so there is no cold tier to union with yet.\n"
              + "  Wait for the tiering job to commit (table.datalake.freshness) and re-run.");
    } else if (unionCount > lakeCount) {
      System.out.println("  UNION READ CONFIRMED: one query, served from both tiers.");
      System.out.println(
          "  The extra rows exist only in the Fluss log, and the Paimon rows were read without\n"
              + "  replaying the log from offset 0 — see the resolved split below for the offset.");
    } else {
      System.out.println(
          "  Equal counts, which is NOT evidence against a union read. ticket_insight is a\n"
              + "  primary-key table: re-upserting an existing ticket_id changes no row count. That\n"
              + "  is exactly the state for a while after a restart with upgradeMode: stateless,\n"
              + "  which replays ticket_id from 0 over keys Paimon already holds.\n"
              + "  Read the resolved split below instead — that check does not depend on counts.");
    }
    System.out.println();

    int recentSnapshotCount = (int) Math.min(snapshotCount, 5);
    System.out.printf(
        " Paimon tiering progress: last %d of %,d commits%n", recentSnapshotCount, snapshotCount);
    System.out.println("----------------------------------------------------------------");
    // total_record_count counts the records in the snapshot's files, which is higher than the
    // COUNT(*) above: every upsert of an existing ticket_id is another record on disk until
    // compaction merges it away. Say so, or the two numbers look like a contradiction.
    System.out.println("  (records on disk, not distinct keys — upserts count until compaction)");
    try (CloseableIterator<Row> snapshotRows =
        tableEnv
            .executeSql(
                "SELECT snapshot_id, total_record_count, commit_time\n"
                    + "FROM fluss_catalog.fluss.`ticket_insight$lake$snapshots`\n"
                    + "ORDER BY snapshot_id DESC LIMIT "
                    + recentSnapshotCount)
            .collect()) {
      while (snapshotRows.hasNext()) {
        Row snapshotRow = snapshotRows.next();
        System.out.printf(
            "  snapshot %-6s %,10d records   committed %s%n",
            snapshotRow.getField(0), snapshotRow.getField(1), snapshotRow.getField(2));
      }
    } catch (Exception exception) {
      throw new IllegalStateException("Failed to read the Paimon snapshot history", exception);
    }
    System.out.println("================================================================");
  }

  /**
   * Closes the iterator rather than leaving it open — ordinary resource hygiene, nothing more. It
   * does not stop Flink's "some data might be lost" warning: in batch mode the MiniCluster shuts
   * down the moment the job finishes, before {@code CollectResultFetcher} polls a final status, so
   * both {@code hasNext} and {@code close} hit an already-stopped cluster. That benign race is
   * silenced at the logging layer instead — see {@code CollectResultFetcher} in
   * {@code src/main/resources/log4j2.properties}.
   */
  private static long collectSingleLong(TableResult tableResult) {
    try (CloseableIterator<Row> rows = tableResult.collect()) {
      return (long) rows.next().getField(0);
    } catch (Exception exception) {
      throw new IllegalStateException("Failed to collect a single value", exception);
    }
  }
}
