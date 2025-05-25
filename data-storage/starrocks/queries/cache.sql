-- Query cache (default: false)
show variables like 'enable_query_cache';
set global enable_query_cache=true;

-- Data cache (default: true)
-- https://docs.starrocks.io/docs/data_source/data_cache/#enable-data-cache
show variables like 'enable_scan_datacache';

-- Footer cache (default: true)
-- https://docs.starrocks.io/docs/data_source/data_cache/#footer-cache
show variables like 'enable_file_metacache';

-- Query timeout (default: 300 seconds)
show variables like 'query_timeout';
set global query_timeout=21600; -- 6 hours
