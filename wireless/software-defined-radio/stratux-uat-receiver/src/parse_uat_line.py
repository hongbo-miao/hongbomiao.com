from models import MessageDirection, RawUatMessage

_DIRECTION_BY_PREFIX: dict[str, MessageDirection] = {
    "-": MessageDirection.DOWNLINK,
    "+": MessageDirection.UPLINK,
}


def parse_uat_line(line: str) -> RawUatMessage | None:
    """Parse one dump978-style text line emitted by the radio.

    Lines look like ``-08abcd...;rs=2;rssi=-21.5;`` where ``-`` marks a
    downlink, ``+`` an uplink, the hex payload follows the prefix, and any
    ``key=value`` metadata trails after the first semicolon.
    """
    line = line.strip()
    if not line or line[0] not in _DIRECTION_BY_PREFIX:
        return None

    direction: MessageDirection = _DIRECTION_BY_PREFIX[line[0]]
    field_list: list[str] = line[1:].split(";")
    hex_payload: str = field_list[0].strip()
    if not hex_payload or len(hex_payload) % 2 != 0:
        return None

    try:
        payload: bytes = bytes.fromhex(hex_payload)
    except ValueError:
        return None

    metadata: dict[str, str] = {}
    for field in field_list[1:]:
        if "=" in field:
            key, value = field.split("=", 1)
            metadata[key.strip()] = value.strip()

    return RawUatMessage(direction=direction, payload=payload, metadata=metadata)
