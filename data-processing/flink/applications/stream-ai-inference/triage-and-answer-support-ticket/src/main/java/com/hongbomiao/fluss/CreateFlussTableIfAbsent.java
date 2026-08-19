package com.hongbomiao.fluss;

import org.apache.flink.table.api.TableEnvironment;
import org.apache.flink.table.catalog.Catalog;
import org.apache.flink.table.catalog.ObjectPath;

/**
 * Issues a {@code CREATE TABLE} only when the table is genuinely absent from the Fluss catalog.
 *
 * <p>SQL's own {@code IF NOT EXISTS} is not sufficient for a {@code table.datalake.enabled} table.
 * The coordinator is supposed to short-circuit on
 * {@code request.isIgnoreIfExists() && metadataManager.tableExists(tablePath)} before it ever
 * touches the lake catalog, but in practice a restart of this job against an already-tiered table
 * still reaches Paimon's {@code checkTableIsEmpty} and dies with "The table fluss.&lt;name&gt;
 * already exists in Paimon catalog, and the table is not empty" — even though the Fluss table is
 * present in ZooKeeper. Since the Paimon tier legitimately outlives any single job run (that is the
 * whole point of tiering), the job could never restart. Checking existence here sidesteps the
 * create request entirely. See README.md's Streamhouse section.
 */
public class CreateFlussTableIfAbsent {

  private CreateFlussTableIfAbsent() {}

  public static void createFlussTableIfAbsent(
      TableEnvironment tableEnv, String catalogName, String databaseName, String tableName, String createTableDdl)
      throws Exception {
    Catalog catalog =
        tableEnv
            .getCatalog(catalogName)
            .orElseThrow(
                () -> new IllegalStateException("Catalog " + catalogName + " is not registered"));
    if (catalog.tableExists(new ObjectPath(databaseName, tableName))) {
      return;
    }
    tableEnv.executeSql(createTableDdl);
  }
}
