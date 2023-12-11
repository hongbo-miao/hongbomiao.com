FROM docker.io/maven:3.9.6-amazoncorretto-11 AS builder
WORKDIR /usr/src/app
COPY ["streaming/src/",  "./src/"]
COPY ["streaming/pom.xml", "./"]
RUN mvn clean package

FROM docker.io/flink:1.18.0-scala_2.12-java11
RUN mkdir -p "${FLINK_HOME}/usrlib"
COPY --from=builder /usr/src/app/target/streaming-0.1.jar "${FLINK_HOME}/usrlib/streaming-0.1.jar"
COPY ["streaming/src/main/resources/application-production.properties", "${FLINK_HOME}/application-production.properties"]
