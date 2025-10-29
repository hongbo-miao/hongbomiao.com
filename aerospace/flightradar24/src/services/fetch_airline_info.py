from fr24sdk.client import Client


def fetch_airline_info(client: Client, airline_icao: str) -> object:
    return client.airlines.get_light(icao=airline_icao)
