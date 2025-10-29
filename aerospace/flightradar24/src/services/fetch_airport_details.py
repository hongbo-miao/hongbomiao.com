from fr24sdk.client import Client


def fetch_airport_details(client: Client, airport_code: str) -> object:
    return client.airports.get_light(code=airport_code)
