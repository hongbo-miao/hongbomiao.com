-- https://clickhouse.com/blog/generating-random-test-distribution-data-for-clickhouse

create table cpu_metrics
(
    `name` String,
    `timestamp` Datetime,
    `value` Float32
)
engine = MERGETREE
order by (name, timestamp)
