FROM docker.io/flink:1.20.5-java11

# Flink 1.20 is not incidental. Paimon's CDC actions, postgres_sync_table among them, only
# exist in the paimon-flink-1.x bundles: paimon-flink-1.20 carries 183 action/cdc classes
# while paimon-flink-2.0, 2.1 and 2.2 carry none, and the paimon-flink-cdc module itself is
# compiled against flink 1.20.1. Moving to Flink 2 means giving up this action entirely and
# switching to a Flink CDC YAML pipeline, which does publish a 2.2 build.
ARG FLINK_MINOR_VERSION=1.20
ARG PAIMON_VERSION=1.4.1
ARG FLINK_CDC_VERSION=3.6.0
ARG FLINK_SHADED_HADOOP_VERSION=2.8.3-10.0
ARG MAVEN_REPOSITORY_URL=https://repo1.maven.org/maven2

# Paimon builds a Hadoop Configuration whenever it opens a catalog, even a filesystem one
# backed by S3, and the Flink image ships no Hadoop at all.
RUN set -eu \
    && curl --silent --show-error --fail --location \
      --output "/opt/flink/lib/flink-shaded-hadoop-2-uber-${FLINK_SHADED_HADOOP_VERSION}.jar" \
      "${MAVEN_REPOSITORY_URL}/org/apache/flink/flink-shaded-hadoop-2-uber/${FLINK_SHADED_HADOOP_VERSION}/flink-shaded-hadoop-2-uber-${FLINK_SHADED_HADOOP_VERSION}.jar"

# The Paimon Flink bundle carries the postgres_sync_table implementation, paimon-s3 gives
# Paimon its own s3:// FileIO for the warehouse, and the Postgres CDC connector is the
# source the action reads from. All three have to be on the cluster classpath.
RUN set -eu \
    && curl --silent --show-error --fail --location \
      --output "/opt/flink/lib/paimon-flink-${FLINK_MINOR_VERSION}-${PAIMON_VERSION}.jar" \
      "${MAVEN_REPOSITORY_URL}/org/apache/paimon/paimon-flink-${FLINK_MINOR_VERSION}/${PAIMON_VERSION}/paimon-flink-${FLINK_MINOR_VERSION}-${PAIMON_VERSION}.jar" \
    && curl --silent --show-error --fail --location \
      --output "/opt/flink/lib/paimon-s3-${PAIMON_VERSION}.jar" \
      "${MAVEN_REPOSITORY_URL}/org/apache/paimon/paimon-s3/${PAIMON_VERSION}/paimon-s3-${PAIMON_VERSION}.jar" \
    && curl --silent --show-error --fail --location \
      --output "/opt/flink/lib/flink-sql-connector-postgres-cdc-${FLINK_CDC_VERSION}-${FLINK_MINOR_VERSION}.jar" \
      "${MAVEN_REPOSITORY_URL}/org/apache/flink/flink-sql-connector-postgres-cdc/${FLINK_CDC_VERSION}-${FLINK_MINOR_VERSION}/flink-sql-connector-postgres-cdc-${FLINK_CDC_VERSION}-${FLINK_MINOR_VERSION}.jar"

# paimon-flink-action is a thin launcher that only holds the FlinkActions main class, so
# it stays out of lib and is submitted as the job jar instead.
RUN set -eu \
    && mkdir --parents /opt/flink/paimon \
    && curl --silent --show-error --fail --location \
      --output /opt/flink/paimon/paimon-flink-action.jar \
      "${MAVEN_REPOSITORY_URL}/org/apache/paimon/paimon-flink-action/${PAIMON_VERSION}/paimon-flink-action-${PAIMON_VERSION}.jar"

# Flink writes its checkpoints through its own filesystem stack, which paimon-s3 does not
# plug into, so the bundled S3 plugin has to be enabled separately.
RUN set -eu \
    && mkdir --parents /opt/flink/plugins/s3-fs-hadoop \
    && cp /opt/flink/opt/flink-s3-fs-hadoop-*.jar /opt/flink/plugins/s3-fs-hadoop/

# Baked into the image rather than passed per pod because the submitter needs the same
# view of the cluster and of checkpointing that the jobmanager and taskmanager have.
# jobmanager.rpc.address is deliberately absent: the entrypoint always rewrites it from
# JOB_MANAGER_RPC_ADDRESS, so it is set as an environment variable instead.
RUN /opt/flink/bin/config-parser-utils.sh /opt/flink/conf /opt/flink/bin /opt/flink/lib \
    -Djobmanager.bind-host=0.0.0.0 \
    -Drest.address=flink-jobmanager \
    -Drest.bind-address=0.0.0.0 \
    -Dtaskmanager.bind-host=0.0.0.0 \
    -Dtaskmanager.numberOfTaskSlots=4 \
    -Dexecution.checkpointing.interval=60s \
    -Dexecution.checkpointing.dir=s3://flink-checkpoints/checkpoints \
    -Ds3.endpoint=http://rustfs-svc:9000 \
    -Ds3.access-key=rustfs_admin \
    -Ds3.secret-key=passw0rd \
    -Ds3.path.style.access=true
