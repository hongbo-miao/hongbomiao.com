FROM docker.io/flink:2.2.1-java17

# Flink CDC publishes each release twice, once per supported Flink minor, so the distribution
# and both pipeline connectors all carry the 3.6.0-2.2 suffix. 2.2 is the newest Flink the
# 3.6.0 line builds against, and there is no 2.3 build of anything here yet.
ARG FLINK_MINOR_VERSION=2.2
ARG FLINK_CDC_VERSION=3.6.0
# Pinned to the Paimon the pipeline connector already bundles (see its paimon-bundle
# pom.properties) so the extra filesystem jar cannot drift from the shaded core.
ARG PAIMON_VERSION=1.3.1
ARG FLINK_SHADED_HADOOP_VERSION=2.8.3-10.0
ARG MAVEN_REPOSITORY_URL=https://repo1.maven.org/maven2
ARG APACHE_ARCHIVE_URL=https://archive.apache.org/dist/flink

# Paimon builds a Hadoop Configuration whenever it opens a catalog, even a filesystem one
# backed by S3, and the Flink image ships no Hadoop at all. This has to sit on the cluster
# classpath rather than beside the connectors, because the shaded Paimon inside the pipeline
# connector resolves it through its parent classloader.
RUN set -eu \
    && curl --silent --show-error --fail --location \
      --output "/opt/flink/lib/flink-shaded-hadoop-2-uber-${FLINK_SHADED_HADOOP_VERSION}.jar" \
      "${MAVEN_REPOSITORY_URL}/org/apache/flink/flink-shaded-hadoop-2-uber/${FLINK_SHADED_HADOOP_VERSION}/flink-shaded-hadoop-2-uber-${FLINK_SHADED_HADOOP_VERSION}.jar"

# The Flink CDC distribution is what actually turns a pipeline YAML into a job graph:
# bin/flink-cdc.sh builds a classpath, hands the YAML to org.apache.flink.cdc.cli.CliFrontend
# and submits the result to an existing cluster over REST.
RUN set -eu \
    && curl --silent --show-error --fail --location \
      --output /tmp/flink-cdc.tar.gz \
      "${APACHE_ARCHIVE_URL}/flink-cdc-${FLINK_CDC_VERSION}/flink-cdc-${FLINK_CDC_VERSION}-${FLINK_MINOR_VERSION}-bin.tar.gz" \
    && mkdir --parents /opt/flink/flink-cdc \
    && tar --extract --gzip --file /tmp/flink-cdc.tar.gz --directory /opt/flink/flink-cdc --strip-components=1 \
    && rm /tmp/flink-cdc.tar.gz

# Pipeline connectors go in the distribution's own lib, not the cluster's: flink-cdc.sh picks
# them up from there and ships them with the job as user jars. paimon-s3 joins them rather
# than the cluster lib so Paimon's s3:// FileIO is registered in the same classloader as the
# Paimon it belongs to, which the bundled loader list (local and Hadoop only) omits.
RUN set -eu \
    && for connector in postgres paimon; do \
      curl --silent --show-error --fail --location \
        --output "/opt/flink/flink-cdc/lib/flink-cdc-pipeline-connector-${connector}-${FLINK_CDC_VERSION}-${FLINK_MINOR_VERSION}.jar" \
        "${MAVEN_REPOSITORY_URL}/org/apache/flink/flink-cdc-pipeline-connector-${connector}/${FLINK_CDC_VERSION}-${FLINK_MINOR_VERSION}/flink-cdc-pipeline-connector-${connector}-${FLINK_CDC_VERSION}-${FLINK_MINOR_VERSION}.jar"; \
    done \
    && curl --silent --show-error --fail --location \
      --output "/opt/flink/flink-cdc/lib/paimon-s3-${PAIMON_VERSION}.jar" \
      "${MAVEN_REPOSITORY_URL}/org/apache/paimon/paimon-s3/${PAIMON_VERSION}/paimon-s3-${PAIMON_VERSION}.jar"

# Flink writes its checkpoints through its own filesystem stack, which paimon-s3 does not
# plug into, so the bundled S3 plugin has to be enabled separately.
RUN set -eu \
    && mkdir --parents /opt/flink/plugins/s3-fs-hadoop \
    && cp /opt/flink/opt/flink-s3-fs-hadoop-*.jar /opt/flink/plugins/s3-fs-hadoop/

# Baked into the image rather than passed per pod because flink-cdc.sh reads this same conf
# directory to find the cluster it should submit to, so the submitter and the cluster have to
# agree. Flink 2 already defaults the bind hosts to 0.0.0.0, but it also defaults rest.address
# to 0.0.0.0, which a submitter cannot dial.
# jobmanager.rpc.address is deliberately absent: the entrypoint always rewrites it from
# JOB_MANAGER_RPC_ADDRESS, so it is set as an environment variable instead.
RUN /opt/flink/bin/config-parser-utils.sh /opt/flink/conf /opt/flink/bin /opt/flink/lib \
    -Drest.address=flink-jobmanager \
    -Drest.bind-address=0.0.0.0 \
    -Dtaskmanager.numberOfTaskSlots=4 \
    -Dexecution.checkpointing.interval=60s \
    -Dexecution.checkpointing.dir=s3://flink-checkpoints/checkpoints \
    -Ds3.endpoint=http://rustfs-svc:9000 \
    -Ds3.access-key=rustfs_admin \
    -Ds3.secret-key=passw0rd \
    -Ds3.path.style.access=true
