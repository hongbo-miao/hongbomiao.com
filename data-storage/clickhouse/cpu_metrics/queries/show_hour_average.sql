-- https://clickhouse.com/blog/generating-random-test-distribution-data-for-clickhouse

select
    TOSTARTOFHOUR(timestamp) as hour,
    ROUND(AVG(value), 2) as val,
    BAR(val, 0, 100) as bar
from cpu_metrics
group by hour
order by hour asc
