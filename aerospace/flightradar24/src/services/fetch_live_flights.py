from fr24sdk.client import Client
from fr24sdk.models.geographic import Boundary


def fetch_live_flights(
    client: Client,
    bounds: Boundary,
    limit_number: int | None = None,
) -> list[object]:
    result = client.live.flight_positions.get_light(bounds=bounds)
    data = getattr(result, "data", []) or []
    if limit_number is not None:
        return data[:limit_number]
    return data
