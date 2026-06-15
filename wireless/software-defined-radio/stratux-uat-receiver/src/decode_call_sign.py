from constants import BASE40_ALPHABET


def decode_call_sign(payload: bytes) -> tuple[str, int]:
    """Decode the base-40 packed call sign and emitter category from a long
    UAT ADS-B frame (DO-282B Mode Status element)."""
    first_word: int = (payload[17] << 8) | payload[18]
    second_word: int = (payload[19] << 8) | payload[20]
    third_word: int = (payload[21] << 8) | payload[22]

    emitter_category: int = (first_word // 1600) % 40
    character_index_list: list[int] = [
        (first_word // 40) % 40,
        first_word % 40,
        (second_word // 1600) % 40,
        (second_word // 40) % 40,
        second_word % 40,
        (third_word // 1600) % 40,
        (third_word // 40) % 40,
        third_word % 40,
    ]
    call_sign: str = "".join(
        BASE40_ALPHABET[index] for index in character_index_list
    ).strip()
    return call_sign, emitter_category
