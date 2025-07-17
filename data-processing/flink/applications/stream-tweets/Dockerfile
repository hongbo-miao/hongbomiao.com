FROM docker.io/maven:3.9.11-amazoncorretto-11 AS builder
WORKDIR /usr/src/app
COPY ["data-processing/flink/applications/stream-tweets/src/",  "./src/"]
COPY ["data-processing/flink/applications/stream-tweets/pom.xml", "./"]
RUN mvn clean package

FROM docker.io/flink:2.0.0-scala_2.12-java11
RUN mkdir -p "${FLINK_HOME}/usrlib"
COPY --from=builder /usr/src/app/target/flink-stream-tweets-0.1.jar "${FLINK_HOME}/usrlib/flink-stream-tweets-0.1.jar"
COPY ["data-processing/flink/applications/stream-tweets/src/main/resources/application-production.properties", "${FLINK_HOME}/application-production.properties"]
