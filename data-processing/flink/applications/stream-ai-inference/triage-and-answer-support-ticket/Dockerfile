FROM docker.io/maven:3.9.11-amazoncorretto-17 AS builder
WORKDIR /usr/src/app
COPY ["src/", "./src/"]
COPY ["pom.xml", "./"]
RUN mvn clean package

FROM docker.io/flink:2.2.1-java17
RUN mkdir -p "${FLINK_HOME}/usrlib"
COPY --from=builder /usr/src/app/target/flink-stream-ai-inference-0.1.jar "${FLINK_HOME}/usrlib/flink-stream-ai-inference-0.1.jar"
COPY ["knowledge-base/", "${FLINK_HOME}/knowledge-base/"]

# Runs as its own FlinkDeployment (kubernetes/flink/fluss-datalake-tiering.yaml) — the datalake
# tiering service is a separate Flink job, not something the streaming job runs in-process.
ADD --chmod=644 https://repo1.maven.org/maven2/org/apache/fluss/fluss-flink-tiering/0.9.1-incubating/fluss-flink-tiering-0.9.1-incubating.jar ${FLINK_HOME}/usrlib/
# Client-side lake connector, needed both by the tiering job above and by any Flink SQL reading a
# $lake / union-read query against a tiered table (e.g. UnionReadReporter).
ADD --chmod=644 https://repo1.maven.org/maven2/org/apache/fluss/fluss-lake-paimon/0.9.1-incubating/fluss-lake-paimon-0.9.1-incubating.jar ${FLINK_HOME}/lib/
# Paimon must be 1.3.1 — the <paimon.version> fluss-lake-paimon 0.9.1-incubating is compiled
# against. Paimon 1.4.x changed CommitCallback.call's signature, so on a 1.4.1 runtime every tiering
# commit dies inside FileStoreCommitImpl.tryCommitOnce with an AbstractMethodError on
# PaimonLakeCommitter$PaimonCommitCallback. See README.md — the symptom looks nothing like a
# version problem.
ADD --chmod=644 https://repo1.maven.org/maven2/org/apache/paimon/paimon-bundle/1.3.1/paimon-bundle-1.3.1.jar ${FLINK_HOME}/lib/
# paimon-bundle alone routes s3:// through Hadoop's S3AFileSystem (needs hadoop-aws + commons-logging,
# not just hadoop-client-*); paimon-s3 is Paimon's own native S3 FileIO and is a separate jar. See
# README.md's Streamhouse section.
ADD --chmod=644 https://repo1.maven.org/maven2/org/apache/paimon/paimon-s3/1.3.1/paimon-s3-1.3.1.jar ${FLINK_HOME}/lib/
# fluss-lake-paimon's Flink-side catalog resolves lake tables via Paimon's Flink integration
# (org.apache.paimon.flink.FlinkFileIOLoader), which paimon-bundle does not include. Paimon 1.3.1
# predates Flink 2.2, so there is no paimon-flink-2.2 at this version; -2.1 is used because staying
# inside one Paimon release is what matters and only the FileIO loader is needed from this jar.
ADD --chmod=644 https://repo1.maven.org/maven2/org/apache/paimon/paimon-flink-2.1/1.3.1/paimon-flink-2.1-1.3.1.jar ${FLINK_HOME}/lib/
# The Flink-client CatalogContext.create() overload Fluss's LakeFlinkCatalog calls eagerly touches
# Hadoop's FileSystem SPI list (org.apache.hadoop.hdfs.HdfsConfiguration, org.apache.hadoop.fs.
# LeaseRecoverable, ...), unlike the Fluss coordinator's own codepath, which doesn't need this. See
# README.md. hadoop-common must be >= 3.4.0 (LeaseRecoverable, HDFS-17094).
ADD --chmod=644 https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-common/3.4.1/hadoop-common-3.4.1.jar ${FLINK_HOME}/lib/
ADD --chmod=644 https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-hdfs-client/3.4.1/hadoop-hdfs-client-3.4.1.jar ${FLINK_HOME}/lib/
# Reading the Paimon tier goes through parquet-hadoop, whose ParquetFileReader references
# mapreduce's FileInputFormat even when no MapReduce job is involved. Read-time only — the catalog
# resolves fine without it, so this surfaces one query later than the jars above.
ADD --chmod=644 https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-mapreduce-client-core/3.4.1/hadoop-mapreduce-client-core-3.4.1.jar ${FLINK_HOME}/lib/

# The Fluss connector is a ~66 MiB shaded jar (bundles fluss-client) that has to sit on Flink's own
# classpath, not in the job jar, so its SPI-registered FlinkCatalogFactory is discoverable.
# --chmod is required here: the base image's final USER is the non-root `flink` user, which can't
# chmod a file ADD just created (owned by root by default), so a separate RUN chmod fails with
# "Operation not permitted".
ADD --chmod=644 https://repo1.maven.org/maven2/org/apache/fluss/fluss-flink-2.2/0.9.1-incubating/fluss-flink-2.2-0.9.1-incubating.jar ${FLINK_HOME}/lib/

# Also needed client-side: reading a Fluss primary-key table means the client opens its KV snapshot
# files itself, directly from `remote.data.dir` (an s3:// URI — see kubernetes/fluss/helm-chart/values.yaml).
ADD --chmod=644 https://repo1.maven.org/maven2/org/apache/fluss/fluss-fs-s3/0.9.1-incubating/fluss-fs-s3-0.9.1-incubating.jar ${FLINK_HOME}/lib/

# Flink's own S3 filesystem, for checkpoint storage (kubernetes/flink/stream-ai-inference.yaml
# state.checkpoints.dir) — a separate client from Fluss's fluss-fs-s3 above, and must go in
# plugins/, not lib/, per Flink's plugin-loading convention.
RUN mkdir -p "${FLINK_HOME}/plugins/s3-fs-hadoop" \
    && cp "${FLINK_HOME}/opt/flink-s3-fs-hadoop-"*.jar "${FLINK_HOME}/plugins/s3-fs-hadoop/"

# Arrow's off-heap buffers reach into java.nio internals that the JDK 17 module system hides by
# default; both the streaming job (VECTOR_SEARCH) and the KnowledgeBaseLoader batch job need this.
RUN /opt/flink/bin/config-parser-utils.sh /opt/flink/conf /opt/flink/bin /opt/flink/lib \
    -Denv.java.opts.all="--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED"
