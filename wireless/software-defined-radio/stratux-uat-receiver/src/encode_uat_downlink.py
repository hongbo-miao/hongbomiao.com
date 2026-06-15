import math

from constants import (
    ALTITUDE_OFFSET_FOOT,
    ALTITUDE_STEP_FOOT,
    BASE40_ALPHABET,
    FULL_CIRCLE_DEGREE,
    LATITUDE_LONGITUDE_COUNT,
    LONG_FRAME_BYTE_COUNT,
    VERTICAL_RATE_STEP_FOOT_PER_MINUTE,
)
from models import UatDownlinkMessage


def _encode_signed_velocity(velocity: int) -> int:
    raw_velocity: int = (abs(velocity) + 1) & 0x3FF
    if velocity < 0:
        raw_velocity |= 0x400
    return raw_velocity


def encode_uat_downlink(message: UatDownlinkMessage) -> bytes:
    """Build a 34-byte long UAT ADS-B frame from a decoded message.

    This is the inverse of ``decode_uat_downlink`` and exists so the demo can
    synthesise frames that exercise the full decode path without live RF.
    """
    payload: bytearray = bytearray(LONG_FRAME_BYTE_COUNT)

    payload[0] = ((message.mdb_type_code & 0x1F) << 3) | (
        int(message.address_qualifier) & 0x07
    )
    address: int = int(message.icao_address, 16)
    payload[1] = (address >> 16) & 0xFF
    payload[2] = (address >> 8) & 0xFF
    payload[3] = address & 0xFF

    latitude_degree: float = message.latitude_degree or 0.0
    longitude_degree: float = message.longitude_degree or 0.0
    if latitude_degree < 0:
        latitude_degree += 180
    if longitude_degree < 0:
        longitude_degree += 360
    raw_latitude: int = (
        round(latitude_degree * LATITUDE_LONGITUDE_COUNT / FULL_CIRCLE_DEGREE) & 0x7FFFFF
    )
    raw_longitude: int = (
        round(longitude_degree * LATITUDE_LONGITUDE_COUNT / FULL_CIRCLE_DEGREE)
        & 0xFFFFFF
    )
    payload[4] = (raw_latitude >> 15) & 0xFF
    payload[5] = (raw_latitude >> 7) & 0xFF
    payload[6] = ((raw_latitude & 0x7F) << 1) | ((raw_longitude >> 23) & 0x01)
    payload[7] = (raw_longitude >> 15) & 0xFF
    payload[8] = (raw_longitude >> 7) & 0xFF
    payload[9] = ((raw_longitude & 0x7F) << 1) | (int(message.altitude_type) & 0x01)

    raw_altitude: int = (
        (message.altitude_foot + ALTITUDE_OFFSET_FOOT) // ALTITUDE_STEP_FOOT + 1
        if message.altitude_foot is not None
        else 0
    )
    payload[10] = (raw_altitude >> 4) & 0xFF
    payload[11] = ((raw_altitude & 0x0F) << 4) | (
        message.navigation_integrity_category & 0x0F
    )

    payload[12] = (int(message.air_ground_state) & 0x03) << 6
    if message.ground_speed_knot is not None and message.track_degree is not None:
        # Reconstruct north/south and east/west components from speed and track.
        track_radian: float = math.radians(message.track_degree)
        north_south_velocity: int = round(message.ground_speed_knot * math.cos(track_radian))
        east_west_velocity: int = round(message.ground_speed_knot * math.sin(track_radian))
        raw_north_south: int = _encode_signed_velocity(north_south_velocity)
        raw_east_west: int = _encode_signed_velocity(east_west_velocity)
        payload[12] |= (raw_north_south >> 6) & 0x1F
        payload[13] = (raw_north_south & 0x3F) << 2
        payload[13] |= (raw_east_west >> 9) & 0x03
        payload[14] = (raw_east_west >> 1) & 0xFF
        payload[15] = (raw_east_west & 0x01) << 7

    if message.vertical_rate_foot_per_minute is not None:
        vertical_rate: int = message.vertical_rate_foot_per_minute
        raw_vertical_rate: int = (
            abs(vertical_rate) // VERTICAL_RATE_STEP_FOOT_PER_MINUTE + 1
        ) & 0x1FF
        if vertical_rate < 0:
            raw_vertical_rate |= 0x200
        payload[15] |= (raw_vertical_rate >> 4) & 0x7F
        payload[16] = (raw_vertical_rate & 0x0F) << 4

    call_sign: str = (message.call_sign or "").ljust(8)[:8]
    character_index_list: list[int] = [
        BASE40_ALPHABET.index(character) if character in BASE40_ALPHABET else 36
        for character in call_sign
    ]
    emitter_category: int = (
        int(message.emitter_category) if message.emitter_category is not None else 0
    )
    first_word: int = (
        emitter_category * 1600 + character_index_list[0] * 40 + character_index_list[1]
    )
    second_word: int = (
        character_index_list[2] * 1600
        + character_index_list[3] * 40
        + character_index_list[4]
    )
    third_word: int = (
        character_index_list[5] * 1600
        + character_index_list[6] * 40
        + character_index_list[7]
    )
    payload[17] = (first_word >> 8) & 0xFF
    payload[18] = first_word & 0xFF
    payload[19] = (second_word >> 8) & 0xFF
    payload[20] = second_word & 0xFF
    payload[21] = (third_word >> 8) & 0xFF
    payload[22] = third_word & 0xFF

    return bytes(payload)
