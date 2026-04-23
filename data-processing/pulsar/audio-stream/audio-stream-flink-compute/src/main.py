import logging
import os

from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.window import SlidingEventTimeWindows, Time
from pyflink.table import StreamTableEnvironment
from shared.audio.services.compute_stream_metrics import ComputeStreamMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

PULSAR_SERVICE_URL = os.environ.get("PULSAR_SERVICE_URL", "pulsar://pulsar:6650")
INPUT_TOPIC = os.environ.get(
    "INPUT_TOPIC",
    "persistent://public/default/audio-transcript",
)
OUTPUT_TOPIC = os.environ.get(
    "OUTPUT_TOPIC",
    "persistent://public/default/audio-transcript-metrics",
)
SUBSCRIPTION_NAME = os.environ.get("SUBSCRIPTION_NAME", "flink-compute-subscription")
PARALLELISM = int(os.environ.get("PARALLELISM", "2"))


def main() -> None:
    logger.info(f"Pulsar service URL: {PULSAR_SERVICE_URL}")
    logger.info(f"Input topic: {INPUT_TOPIC}")
    logger.info(f"Output topic: {OUTPUT_TOPIC}")
    logger.info(f"Parallelism: {PARALLELISM}")

    environment = StreamExecutionEnvironment.get_execution_environment()
    environment.set_parallelism(PARALLELISM)
    environment.enable_checkpointing(60_000)

    table_environment = StreamTableEnvironment.create(environment)

    table_environment.execute_sql(f"""
        create table pulsar_transcript (
            `device_id` string not null,
            `text` string not null,
            `timestamp_ns` bigint not null,
            `event_time` as to_timestamp_ltz(`timestamp_ns` / 1000000, 3),
            watermark for `event_time` as `event_time` - interval '3' second
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

    transcript_table = table_environment.from_path("pulsar_transcript")
    transcript_stream = table_environment.to_data_stream(transcript_table)

    metric_stream = (
        transcript_stream.key_by(lambda row: row[0])
        .window(SlidingEventTimeWindows.of(Time.seconds(10), Time.seconds(2)))
        .process(ComputeStreamMetrics(), output_type=Types.STRING())
    )

    metric_stream.print()
    environment.execute("audio-stream-flink-compute")


if __name__ == "__main__":
    main()
