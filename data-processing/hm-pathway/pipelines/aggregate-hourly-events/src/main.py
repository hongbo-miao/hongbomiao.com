import logging
from pathlib import Path

import pathway as pw

logger = logging.getLogger(__name__)


def create_sample_event_data() -> pw.Table:
    # 1704103200000000000 = 2024-01-01 10:00:00 UTC
    # 1704104100000000000 = 2024-01-01 10:15:00 UTC
    # 1704105900000000000 = 2024-01-01 10:45:00 UTC
    # 1704107400000000000 = 2024-01-01 11:10:00 UTC
    # 1704108600000000000 = 2024-01-01 11:30:00 UTC
    # 1704110700000000000 = 2024-01-01 12:05:00 UTC
    return pw.debug.table_from_markdown(
        """
            | event_time          | value
        1   | 1704103200000000000 | 10
        2   | 1704104100000000000 | 20
        3   | 1704105900000000000 | 30
        4   | 1704107400000000000 | 40
        5   | 1704108600000000000 | 50
        6   | 1704110700000000000 | 60
        """,
    )


def parse_event_time(event_data: pw.Table) -> pw.Table:
    return event_data.with_columns(
        event_time=pw.this.event_time.dt.from_timestamp(unit="ns"),
    )


def aggregate_by_tumbling_window(event_data: pw.Table) -> pw.Table:
    return event_data.windowby(
        event_data.event_time,
        window=pw.temporal.tumbling(duration=pw.Duration("1h")),
    ).reduce(
        window_start=pw.cast(int, pw.this._pw_window_start.dt.timestamp(unit="ns")),  # noqa: SLF001
        window_end=pw.cast(int, pw.this._pw_window_end.dt.timestamp(unit="ns")),  # noqa: SLF001
        event_count=pw.reducers.count(),
        total_value=pw.reducers.sum(pw.this.value),
    )


def main() -> None:
    event_data = create_sample_event_data()
    event_data = parse_event_time(event_data)
    results = aggregate_by_tumbling_window(event_data)
    pw.io.jsonlines.write(results, Path("output/output.jsonl"))
    pw.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
