from utils.iads_util import IadsUtil


class TestParseMarkers:
    def test_parse_markers(self) -> None:
        markers = IadsUtil.parse_markers(
            (
                "User1|001:01:00:00.100|Marker1|\x00",
                "User1|001:01:00:00.200|Marker2|\x00",
            ),
            2025,
        )
        assert markers == [
            {
                "time": 1735779600100000000,
                "comment": "Marker1",
                "user": "User1",
                "property_bag": "",
            },
            {
                "time": 1735779600200000000,
                "comment": "Marker2",
                "user": "User1",
                "property_bag": "",
            },
        ]


class TestParseTestPoints:
    def test_parse_test_points(self) -> None:
        test_points = IadsUtil.parse_test_points(
            (
                "User1||TestPoint1||001:01:00:00.100|001:01:00:00.200|\x00",
                "User1||TestPoint2||001:01:00:00.300|001:01:00:00.400|\x00",
            ),
            2025,
        )
        assert test_points == [
            {
                "user": "User1",
                "test_point": "",
                "description": "TestPoint1",
                "maneuver": "",
                "start_time": 1735779600100000000,
                "end_time": 1735779600200000000,
                "property_bag": "",
            },
            {
                "user": "User1",
                "test_point": "",
                "description": "TestPoint2",
                "maneuver": "",
                "start_time": 1735779600300000000,
                "end_time": 1735779600400000000,
                "property_bag": "",
            },
        ]
