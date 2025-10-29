import logging
import os

from config import config
from fr24sdk.client import Client
from fr24sdk.exceptions import AuthenticationError, Fr24SdkError
from fr24sdk.models.geographic import Boundary
from services.check_usage_statistics import check_usage_statistics
from services.fetch_airline_info import fetch_airline_info
from services.fetch_airport_details import fetch_airport_details
from services.fetch_live_flights import fetch_live_flights

logger = logging.getLogger(__name__)


def main() -> None:
    os.environ["FR24_API_TOKEN"] = config.FLIGHTRADAR24_API_TOKEN

    try:
        with Client() as flightradar24_client:
            # Airports
            logger.info("Fetching airport details")
            airport_code = "JFK"
            airports = fetch_airport_details(flightradar24_client, airport_code)
            logger.info(f"{airports = }")

            # Airlines
            logger.info("Fetching airline information")
            airline_icao = "DAL"
            airlines = fetch_airline_info(flightradar24_client, airline_icao)
            logger.info(f"{airlines = }")

            # Live flights
            logger.info("Fetching live flights in area")
            bounds = Boundary(north=41.0, south=40.0, west=-74.5, east=-73.0)
            limit_number = 5
            flights = fetch_live_flights(
                flightradar24_client,
                bounds=bounds,
                limit_number=limit_number,
            )
            logger.info(f"{flights = }")

            # Usage statistics
            logger.info("Fetching API usage statistics")
            usage = check_usage_statistics(flightradar24_client)
            if usage is None:
                logger.warning("No usage data available")
            else:
                logger.info(f"{usage = }")

    except AuthenticationError:
        logger.exception("Authentication failed")
    except Fr24SdkError:
        logger.exception("SDK Error")
    except Exception:
        logger.exception("Unexpected error")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
