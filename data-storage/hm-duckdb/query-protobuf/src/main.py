import logging

import duckdb


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
        logging.info(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
