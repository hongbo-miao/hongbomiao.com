import logging

import duckdb

logger = logging.getLogger(__name__)


def main() -> None:
    with duckdb.connect(config={"allow_unsigned_extensions": True}) as conn:
        conn.load_extension("protobuf")
        query = """
            select *
            from protobuf(
                descriptors = './data/motor_descriptor.pb',
                files = './data/motor_data.pb',
                message_type = 'production.iot.Motor',
                delimiter = 'BigEndianFixed'
            )
            order by timestamp_ns
        """
        df = conn.execute(query).pl()
        logger.info(df)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
