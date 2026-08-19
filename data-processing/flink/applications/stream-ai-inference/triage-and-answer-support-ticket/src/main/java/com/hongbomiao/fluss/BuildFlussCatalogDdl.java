package com.hongbomiao.fluss;

/**
 * Builds the {@code CREATE CATALOG} statement shared by the streaming job and the union-read
 * reporter.
 *
 * <p>The {@code paimon.*} options are what let this client reach the Paimon tier of a
 * datalake-enabled table. They are <em>not</em> inherited from the coordinator the way
 * {@code remote.data.dir}'s credentials are: {@code FlinkCatalogFactory} builds its
 * {@code lakeCatalogProperties} from {@code context.getOptions()} — the options in this very DDL —
 * by extracting everything prefixed with the lake format's name. Without them, resolving a
 * {@code $lake} table fails with {@code UnsupportedSchemeException: Could not find a file io
 * implementation for scheme 's3'}. See README.md's Streamhouse section.
 *
 * <p>Only {@code bootstrap.servers}, {@code default-database} and {@code property-version} are
 * validated; {@code paimon.*} (like {@code client.security.*}) is in the factory's
 * skip-validation prefix list, which is why these are accepted here while a bare {@code s3.*} is
 * rejected.
 */
public class BuildFlussCatalogDdl {

  private BuildFlussCatalogDdl() {}

  public static String buildFlussCatalogDdl(
      String flussBootstrapServers, String s3Endpoint, String s3AccessKey, String s3SecretKey) {
    return String.format(
        "CREATE CATALOG fluss_catalog WITH (\n"
            + "  'type' = 'fluss',\n"
            + "  'bootstrap.servers' = '%s',\n"
            + "  'paimon.metastore' = 'filesystem',\n"
            + "  'paimon.warehouse' = 's3://paimon-warehouse/fluss',\n"
            + "  'paimon.s3.endpoint' = '%s',\n"
            + "  'paimon.s3.access-key' = '%s',\n"
            + "  'paimon.s3.secret-key' = '%s',\n"
            + "  'paimon.s3.path.style.access' = 'true'\n"
            + ")",
        flussBootstrapServers, s3Endpoint, s3AccessKey, s3SecretKey);
  }
}
