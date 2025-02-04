import logging

import polars as pl
from pyarrow import flight

logger = logging.getLogger(__name__)


def main() -> None:
    # Connect to the Flight SQL server
    client = flight.connect("grpc://localhost:50841")

    # List all available flights
    flights = client.list_flights()
    logger.info("Available flights:")
    for flight_info in flights:
        logger.info(f"Path: {flight_info.descriptor.path}")
        logger.info(f"Total records: {flight_info.total_records}")
        logger.info(f"Total bytes: {flight_info.total_bytes}")

    descriptor = flight.FlightDescriptor.for_path("data.parquet")

    # Get the flight info
    flight_info = client.get_flight_info(descriptor)

    # Get the data
    reader = client.do_get(flight_info.endpoints[0].ticket)

    # Read all the batches as arrow table and convert to polars
    arrow_table = reader.read_all()
    df = pl.from_arrow(arrow_table)

    # Select specific columns
    result_df = df.select(
        [
            "timestamp",
            "current",
            "voltage",
            "temperature",
        ],
    )
    logger.info(result_df)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
