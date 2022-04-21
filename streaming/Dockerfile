FROM flink:1.14.4-scala_2.12-java11

RUN mkdir -p "${FLINK_HOME}/usrlib"
COPY ["streaming/src/main/resources/application-production.properties", "${FLINK_HOME}/application-production.properties"]
COPY ["streaming/target/streaming-0.1.jar", "${FLINK_HOME}/usrlib/streaming-0.1.jar"]
