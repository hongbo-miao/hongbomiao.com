-- https://clickhouse.com/blog/generating-random-test-distribution-data-for-clickhouse

create table cpu_metrics
(
    `name` String,
    `timestamp` datetime,
    `value` Float32
)
    engine = MergeTree
        order by (name, timestamp)
