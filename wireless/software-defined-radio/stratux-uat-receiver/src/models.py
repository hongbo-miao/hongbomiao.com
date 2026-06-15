from enum import IntEnum, StrEnum

from pydantic import BaseModel


class MessageDirection(StrEnum):
    DOWNLINK = "downlink"
    UPLINK = "uplink"


class AddressQualifier(IntEnum):
    ADSB_WITH_ICAO_ADDRESS = 0
    ADSB_WITH_SELF_ASSIGNED_ADDRESS = 1
    TISB_WITH_ICAO_ADDRESS = 2
    TISB_WITH_TRACK_FILE_ADDRESS = 3
    SURFACE_VEHICLE = 4
    FIXED_GROUND_BEACON = 5
    RESERVED_SIX = 6
    RESERVED_SEVEN = 7


class AirGroundState(IntEnum):
    AIRBORNE_SUBSONIC = 0
    AIRBORNE_SUPERSONIC = 1
    ON_GROUND = 2
    RESERVED = 3


class AltitudeType(IntEnum):
    BAROMETRIC = 0
    GEOMETRIC = 1


class EmitterCategory(IntEnum):
    NO_INFORMATION = 0
    LIGHT = 1
    SMALL = 2
    LARGE = 3
    HIGH_VORTEX_LARGE = 4
    HEAVY = 5
    HIGHLY_MANEUVERABLE = 6
    ROTORCRAFT = 7
    GLIDER_OR_SAILPLANE = 9
    LIGHTER_THAN_AIR = 10
    PARACHUTIST = 11
    ULTRALIGHT = 12
    UNMANNED_AERIAL_VEHICLE = 14
    SPACE_VEHICLE = 15
    SURFACE_EMERGENCY_VEHICLE = 17
    SURFACE_SERVICE_VEHICLE = 18
    POINT_OBSTACLE = 19

    @classmethod
    def _missing_(cls, value: object) -> "EmitterCategory":
        return cls.NO_INFORMATION


class RawUatMessage(BaseModel):
    direction: MessageDirection
    payload: bytes
    metadata: dict[str, str]


class UatDownlinkMessage(BaseModel):
    mdb_type_code: int
    address_qualifier: AddressQualifier
    icao_address: str
    latitude_degree: float | None
    longitude_degree: float | None
    altitude_foot: int | None
    altitude_type: AltitudeType
    navigation_integrity_category: int
    air_ground_state: AirGroundState
    ground_speed_knot: int | None
    track_degree: int | None
    vertical_rate_foot_per_minute: int | None
    call_sign: str | None
    emitter_category: EmitterCategory | None
