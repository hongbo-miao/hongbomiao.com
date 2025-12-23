import re

FLOAT_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def parse_segment(segment: str) -> tuple[float, float, str]:
    float_matches = FLOAT_PATTERN.findall(segment)
    if len(float_matches) < 2:
        msg = f"Segment lacks numeric timestamps: {segment}"
        raise ValueError(msg)

    speaker_identifier = segment.split()[-1]
    return float(float_matches[0]), float(float_matches[1]), speaker_identifier
