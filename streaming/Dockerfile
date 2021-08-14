FROM flink:1.13.2-scala_2.12-java11

RUN mkdir -p "${FLINK_HOME}/usrlib"
COPY ["streaming/target/streaming-0.1.jar", "${FLINK_HOME}/usrlib/streaming-0.1.jar"]
