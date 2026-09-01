-- One query, two answers -- Ranger rewrites what Doris actually runs.
-- application_a_user: ~400K rows (device_id <= 100), 100 devices, NULL checksum (reading masked).
-- application_b_user: the full 2M / 500 / real checksum.
select count(*) as row_count,
    count(distinct device_id) as device_count,
    round(sum(reading), 6) as reading_checksum
from paimon_catalog.sensor_db.sensor_reading;
