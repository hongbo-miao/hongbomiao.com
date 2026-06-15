import time
from collections.abc import Iterator

from encode_uat_downlink import encode_uat_downlink
from models import (
    AddressQualifier,
    AirGroundState,
    AltitudeType,
    EmitterCategory,
    MessageDirection,
    RawUatMessage,
    UatDownlinkMessage,
)

_SAMPLE_AIRCRAFT_LIST: list[UatDownlinkMessage] = [
    UatDownlinkMessage(
        mdb_type_code=1,
        address_qualifier=AddressQualifier.ADSB_WITH_ICAO_ADDRESS,
        icao_address="A1B2C3",
        latitude_degree=37.6213,
        longitude_degree=-122.3790,
        altitude_foot=4500,
        altitude_type=AltitudeType.BAROMETRIC,
        navigation_integrity_category=8,
        air_ground_state=AirGroundState.AIRBORNE_SUBSONIC,
        ground_speed_knot=145,
        track_degree=90,
        vertical_rate_foot_per_minute=640,
        call_sign="N172SP",
        emitter_category=EmitterCategory.LIGHT,
    ),
    UatDownlinkMessage(
        mdb_type_code=1,
        address_qualifier=AddressQualifier.ADSB_WITH_ICAO_ADDRESS,
        icao_address="ABCDEF",
        latitude_degree=40.6895,
        longitude_degree=-74.1745,
        altitude_foot=2750,
        altitude_type=AltitudeType.GEOMETRIC,
        navigation_integrity_category=9,
        air_ground_state=AirGroundState.AIRBORNE_SUBSONIC,
        ground_speed_knot=88,
        track_degree=215,
        vertical_rate_foot_per_minute=-512,
        call_sign="N99HM",
        emitter_category=EmitterCategory.ROTORCRAFT,
    ),
    UatDownlinkMessage(
        mdb_type_code=1,
        address_qualifier=AddressQualifier.ADSB_WITH_ICAO_ADDRESS,
        icao_address="C0FFEE",
        latitude_degree=51.4700,
        longitude_degree=-0.4543,
        altitude_foot=33000,
        altitude_type=AltitudeType.BAROMETRIC,
        navigation_integrity_category=10,
        air_ground_state=AirGroundState.AIRBORNE_SUBSONIC,
        ground_speed_knot=420,
        track_degree=305,
        vertical_rate_foot_per_minute=0,
        call_sign="HEAVY1",
        emitter_category=EmitterCategory.HEAVY,
    ),
]


def simulate_uat_stream(
    message_interval_second: float = 1.0,
) -> Iterator[RawUatMessage]:
    """Replay synthetic downlink frames so the demo runs without the radio.

    Each sample aircraft is encoded into a real UAT frame and re-emitted as a
    raw message, exercising the same decode path as live RF.
    """
    while True:
        for aircraft in _SAMPLE_AIRCRAFT_LIST:
            yield RawUatMessage(
                direction=MessageDirection.DOWNLINK,
                payload=encode_uat_downlink(aircraft),
                metadata={"rs": "0", "rssi": "-22.0", "source": "simulated"},
            )
            time.sleep(message_interval_second)
