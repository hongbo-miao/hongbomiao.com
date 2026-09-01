# Pinned to 2.8.0 to match the ranger-plugins-common the Doris FE compiles against.
FROM docker.io/apache/ranger:2.8.0

# Never merged into Apache Ranger, so neither jar ships with the image or exists on Maven Central;
# these are the URLs the Doris documentation points at. Server-side only: they back the UI's Test
# Connection and autocomplete, which reach Doris over MySQL, hence the JDBC driver.
ARG RANGER_DORIS_PLUGIN_JAR_URL=https://selectdb-doris-1308700295.cos.ap-beijing.myqcloud.com/release/ranger/dev/ranger-doris-plugin-3.0.0-SNAPSHOT.jar
ARG MYSQL_CONNECTOR_JAR_URL=https://selectdb-doris-1308700295.cos.ap-beijing.myqcloud.com/release/jdbc_driver/mysql-connector-java-8.0.25.jar
ARG RANGER_DORIS_PLUGIN_DIRECTORY=/opt/ranger/admin/ews/webapp/WEB-INF/classes/ranger-plugins/doris

RUN set -eu \
    && mkdir --parents "${RANGER_DORIS_PLUGIN_DIRECTORY}" \
    && curl --silent --show-error --fail --location \
      --output "${RANGER_DORIS_PLUGIN_DIRECTORY}/ranger-doris-plugin.jar" \
      "${RANGER_DORIS_PLUGIN_JAR_URL}" \
    && curl --silent --show-error --fail --location \
      --output "${RANGER_DORIS_PLUGIN_DIRECTORY}/mysql-connector-java.jar" \
      "${MYSQL_CONNECTOR_JAR_URL}"
