-- Query cache (default: false)
show variables where variable_name = 'enable_query_cache';
set global enable_query_cache=true;

-- Data cache (default: true)
-- https://docs.starrocks.io/docs/data_source/data_cache/#enable-data-cache
show variables where variable_name = 'enable_scan_datacache';

-- Footer cache (default: true)
-- https://docs.starrocks.io/docs/data_source/data_cache/#footer-cache
show variables where variable_name = 'enable_file_metacache';
