from datetime import datetime

from fr24sdk.client import Client


def fetch_flight_summary(
    client: Client,
    flight_id: str,
    from_time: datetime,
    to_time: datetime,
) -> object:
    return client.flight_summary.get_light(
        flights=[flight_id],
        flight_datetime_from=from_time,
        flight_datetime_to=to_time,
    )
