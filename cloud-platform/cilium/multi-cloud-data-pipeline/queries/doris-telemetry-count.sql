-- Same warehouse each cloud's own Flink job writes to -- Doris reads it as an external
-- catalog, matching each cloud's local warehouse only, never the other cloud's.
--
-- refresh first: Doris caches the Paimon snapshot it last planned against, and a concurrent
-- Flink writer can expire that snapshot via compaction before Doris re-reads it, which then
-- fails the query with "scan snapshotId is out of available snapshotId range" rather than
-- transparently picking up the newer one.
refresh catalog paimon_catalog;

select count(*) as row_count, count(distinct device_id) as device_count
from paimon_catalog.telemetry.sensor_reading;
