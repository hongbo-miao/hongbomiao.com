-- https://clickhouse.com/blog/generating-random-test-distribution-data-for-clickhouse

select
    toStartOfHour(timestamp) as hour,
    round(avg(value), 2) as val,
    bar(val, 0, 100)
from cpu_metrics
group by hour
order by hour asc
