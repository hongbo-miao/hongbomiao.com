from decode_uat_downlink import decode_uat_downlink
from encode_uat_downlink import encode_uat_downlink
from models import (
    AddressQualifier,
    AirGroundState,
    AltitudeType,
    EmitterCategory,
    MessageDirection,
    UatDownlinkMessage,
)
from parse_uat_line import parse_uat_line


def _build_sample_message() -> UatDownlinkMessage:
    return UatDownlinkMessage(
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
    )


def test_encode_decode_round_trip_preserves_identity() -> None:
    original = _build_sample_message()
    decoded = decode_uat_downlink(encode_uat_downlink(original))

    assert decoded.icao_address == original.icao_address
    assert decoded.address_qualifier == original.address_qualifier
    assert decoded.call_sign == original.call_sign
    assert decoded.emitter_category == original.emitter_category
    assert decoded.altitude_type == original.altitude_type
    assert decoded.air_ground_state == original.air_ground_state


def test_encode_decode_round_trip_preserves_position_within_tolerance() -> None:
    original = _build_sample_message()
    decoded = decode_uat_downlink(encode_uat_downlink(original))

    assert abs(decoded.latitude_degree - original.latitude_degree) < 0.001
    assert abs(decoded.longitude_degree - original.longitude_degree) < 0.001
    assert abs(decoded.altitude_foot - original.altitude_foot) <= 25
    assert abs(decoded.ground_speed_knot - original.ground_speed_knot) <= 2
    assert abs(decoded.track_degree - original.track_degree) <= 2


def test_parse_uat_line_reads_downlink_and_metadata() -> None:
    message = parse_uat_line("-0102030405060708090a0b0c0d0e0f1011;rs=2;rssi=-21.5;")

    assert message is not None
    assert message.direction == MessageDirection.DOWNLINK
    assert len(message.payload) == 17
    assert message.metadata["rs"] == "2"
    assert message.metadata["rssi"] == "-21.5"


def test_parse_uat_line_rejects_garbage() -> None:
    assert parse_uat_line("") is None
    assert parse_uat_line("not a frame") is None
    assert parse_uat_line("-xyz;") is None
