import logging
import os

from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment
from shared.telemetry.services.detect_batch_boundary import DetectBatchBoundary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

PULSAR_SERVICE_URL = os.environ.get("PULSAR_SERVICE_URL", "pulsar://pulsar:6650")
INPUT_TOPIC = os.environ.get(
    "INPUT_TOPIC",
    "persistent://public/default/sensor-telemetry-streams",
)
OUTPUT_TOPIC = os.environ.get(
    "OUTPUT_TOPIC",
    "persistent://public/default/sensor-telemetry-batch-results",
)
SUBSCRIPTION_NAME = os.environ.get("SUBSCRIPTION_NAME", "flink-compute-subscription")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "10"))
PARALLELISM = int(os.environ.get("PARALLELISM", "2"))
CONNECTOR_JAR_PATH = os.environ.get(
    "CONNECTOR_JAR_PATH",
    "/app/lib/flink-sql-connector-pulsar-4.1.0-1.18.jar",
)
AVRO_JAR_PATH = os.environ.get(
    "AVRO_JAR_PATH",
    "/app/lib/flink-sql-avro-1.20.0.jar",
)


def main() -> None:
    logger.info(f"Pulsar service URL: {PULSAR_SERVICE_URL}")
    logger.info(f"Input topic: {INPUT_TOPIC}")
    logger.info(f"Output topic: {OUTPUT_TOPIC}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Parallelism: {PARALLELISM}")

    environment = StreamExecutionEnvironment.get_execution_environment()
    environment.set_parallelism(PARALLELISM)
    environment.enable_checkpointing(60_000)
    environment.add_jars(
        f"file://{CONNECTOR_JAR_PATH}",
        f"file://{AVRO_JAR_PATH}",
    )

    table_environment = StreamTableEnvironment.create(environment)

    table_environment.execute_sql(f"""
        create table pulsar_telemetry (
            `publisher_id` string,
            `timestamp_ns` bigint,
            `temperature_c` double,
            `humidity_pct` double
        ) with (
            'connector' = 'pulsar',
            'topics' = '{INPUT_TOPIC}',
            'service-url' = '{PULSAR_SERVICE_URL}',
            'source.subscription-name' = '{SUBSCRIPTION_NAME}',
            'source.subscription-type' = 'Key_Shared',
            'source.start.message-id' = 'earliest',
            'value.format' = 'avro'
        )
    """)

    telemetry_table = table_environment.from_path("pulsar_telemetry")
    telemetry_stream = table_environment.to_data_stream(telemetry_table)

    batch_stream = telemetry_stream.key_by(lambda row: row[0]).process(
        DetectBatchBoundary(
            batch_size=BATCH_SIZE,
            pulsar_service_url=PULSAR_SERVICE_URL,
            output_topic=OUTPUT_TOPIC,
        ),
        output_type=Types.STRING(),
    )

    batch_stream.print()
    environment.execute("Flink Telemetry Batch Compute")


if __name__ == "__main__":
    main()
