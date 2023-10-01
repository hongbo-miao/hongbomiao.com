-- https://clickhouse.com/blog/generating-random-test-distribution-data-for-clickhouse

insert into cpu_metrics
select
    'cpu',
    ts + ((60.0 * 60.0) * randCanonical()) as ts,
    round(val * (0.95 + (randCanonical() / 20.0)), 2) as val
from (
    select
        toDateTime('2022-12-12 12:00:00') - interval h hour as ts,
        round((100.0 * count) / m, 2) as val
    from (
        select
            h,
            count,
            max(count) over () as m
        from (
            select
                floor(randBinomial(24, 0.5) - 12) as h,
                count(*) as count
            from numbers(1000)
            group by h
            order by h asc
        )
    )
) as a
inner join numbers(1000000) as b on 1 = 1
