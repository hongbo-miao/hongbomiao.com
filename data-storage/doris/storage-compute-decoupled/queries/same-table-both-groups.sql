-- Run via just same-table-both-groups-application-a and just same-table-both-groups-application-b.
-- Identical triples from two disjoint compute groups over one Paimon table, zero copies. The
-- checksum is the load-bearing column: matching counts could be coincidence, a matching float
-- sum over 2M rows cannot.
select count(*) as row_count,
    count(distinct device_id) as device_count,
    round(sum(reading), 6) as reading_checksum
from paimon_catalog.sensor_db.sensor_reading;
