-- https://clickhouse.com/blog/generating-random-test-distribution-data-for-clickhouse

insert into cpu_metrics
select
    'cpu' as name,
    ts + ((60.0 * 60.0) * RANDCANONICAL()) as ts,
    ROUND(val * (0.95 + (RANDCANONICAL() / 20.0)), 2) as val
from (
    select
        TODATETIME('2022-12-12 12:00:00') - interval h hour as ts,
        ROUND((100.0 * count) / m, 2) as val
    from (
        select
            h,
            count,
            MAX(count) over () as m
        from (
            select
                FLOOR(RANDBINOMIAL(24, 0.5) - 12) as h,
                COUNT(*) as count
            from NUMBERS(1000)
            group by h
            order by h asc
        )
    )
) as a
inner join NUMBERS(1000000) on 1 = 1
