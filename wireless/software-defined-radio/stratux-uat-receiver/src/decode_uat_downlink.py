import math

from constants import (
    ALTITUDE_OFFSET_FOOT,
    ALTITUDE_STEP_FOOT,
    FULL_CIRCLE_DEGREE,
    LATITUDE_LONGITUDE_COUNT,
    LONG_FRAME_BYTE_COUNT,
    SUPERSONIC_VELOCITY_MULTIPLIER,
    VERTICAL_RATE_STEP_FOOT_PER_MINUTE,
)
from decode_call_sign import decode_call_sign
from models import (
    AddressQualifier,
    AirGroundState,
    AltitudeType,
    EmitterCategory,
    UatDownlinkMessage,
)


def _decode_signed_velocity(
    raw_velocity: int,
    air_ground_state: AirGroundState,
) -> int:
    velocity: int = (raw_velocity & 0x3FF) - 1
    if raw_velocity & 0x400:
        velocity = -velocity
    if air_ground_state == AirGroundState.AIRBORNE_SUPERSONIC:
        velocity *= SUPERSONIC_VELOCITY_MULTIPLIER
    return velocity


def decode_uat_downlink(payload: bytes) -> UatDownlinkMessage:
    """Decode a UAT ADS-B downlink message following DO-282B and the
    cyoung/stratux dump978 reference implementation."""
    mdb_type_code: int = (payload[0] >> 3) & 0x1F
    address_qualifier: AddressQualifier = AddressQualifier(payload[0] & 0x07)
    icao_address: str = f"{(payload[1] << 16) | (payload[2] << 8) | payload[3]:06X}"

    raw_latitude: int = (payload[4] << 15) | (payload[5] << 7) | (payload[6] >> 1)
    raw_longitude: int = (
        ((payload[6] & 0x01) << 23)
        | (payload[7] << 15)
        | (payload[8] << 7)
        | (payload[9] >> 1)
    )
    latitude_degree: float = raw_latitude * FULL_CIRCLE_DEGREE / LATITUDE_LONGITUDE_COUNT
    if latitude_degree > 90:
        latitude_degree -= 180
    longitude_degree: float = (
        raw_longitude * FULL_CIRCLE_DEGREE / LATITUDE_LONGITUDE_COUNT
    )
    if longitude_degree > 180:
        longitude_degree -= 360

    raw_altitude: int = (payload[10] << 4) | ((payload[11] & 0xF0) >> 4)
    altitude_foot: int | None = (
        (raw_altitude - 1) * ALTITUDE_STEP_FOOT - ALTITUDE_OFFSET_FOOT
        if raw_altitude != 0
        else None
    )
    altitude_type: AltitudeType = (
        AltitudeType.GEOMETRIC if payload[9] & 1 else AltitudeType.BAROMETRIC
    )
    navigation_integrity_category: int = payload[11] & 0x0F

    air_ground_state: AirGroundState = AirGroundState((payload[12] >> 6) & 0x03)
    ground_speed_knot: int | None = None
    track_degree: int | None = None
    vertical_rate_foot_per_minute: int | None = None
    if air_ground_state in (
        AirGroundState.AIRBORNE_SUBSONIC,
        AirGroundState.AIRBORNE_SUPERSONIC,
    ):
        raw_north_south: int = ((payload[12] & 0x1F) << 6) | ((payload[13] & 0xFC) >> 2)
        raw_east_west: int = (
            ((payload[13] & 0x03) << 9)
            | (payload[14] << 1)
            | ((payload[15] & 0x80) >> 7)
        )
        north_south_velocity: int = _decode_signed_velocity(
            raw_north_south, air_ground_state
        )
        east_west_velocity: int = _decode_signed_velocity(
            raw_east_west, air_ground_state
        )
        ground_speed_knot = int(
            math.sqrt(
                north_south_velocity * north_south_velocity
                + east_west_velocity * east_west_velocity
            )
        )
        track_degree = (
            int(
                360
                + 90
                - math.atan2(north_south_velocity, east_west_velocity) * 180 / math.pi
            )
            % 360
        )

        raw_vertical_rate: int = ((payload[15] & 0x7F) << 4) | ((payload[16] & 0xF0) >> 4)
        vertical_rate_foot_per_minute = (
            (raw_vertical_rate & 0x1FF) - 1
        ) * VERTICAL_RATE_STEP_FOOT_PER_MINUTE
        if raw_vertical_rate & 0x200:
            vertical_rate_foot_per_minute = -vertical_rate_foot_per_minute

    call_sign: str | None = None
    emitter_category: EmitterCategory | None = None
    if len(payload) >= LONG_FRAME_BYTE_COUNT:
        call_sign, raw_emitter_category = decode_call_sign(payload)
        emitter_category = EmitterCategory(raw_emitter_category)

    return UatDownlinkMessage(
        mdb_type_code=mdb_type_code,
        address_qualifier=address_qualifier,
        icao_address=icao_address,
        latitude_degree=latitude_degree,
        longitude_degree=longitude_degree,
        altitude_foot=altitude_foot,
        altitude_type=altitude_type,
        navigation_integrity_category=navigation_integrity_category,
        air_ground_state=air_ground_state,
        ground_speed_knot=ground_speed_knot,
        track_degree=track_degree,
        vertical_rate_foot_per_minute=vertical_rate_foot_per_minute,
        call_sign=call_sign,
        emitter_category=emitter_category,
    )
