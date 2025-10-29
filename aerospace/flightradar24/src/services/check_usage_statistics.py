from fr24sdk.client import Client


def check_usage_statistics(client: Client) -> object:
    return client.usage.get()
