with t2 as (
    with t1 as (
        select
            pulocationid,
            dolocationid,
            count(*) as total
        from trips
        group by pulocationid, dolocationid
    )

    select
        t1.pulocationid,
        zones.zone as pulocation_zone,
        zones.borough as pulocation_borough,
        t1.dolocationid,
        t1.total
    from t1
    inner join zones on t1.pulocationid = zones.locationid
)

select
    t2.pulocationid,
    t2.pulocation_zone,
    t2.pulocation_borough,
    t2.dolocationid,
    zones.zone as dolocation_zone,
    zones.borough as dolocation_borough,
    t2.total
from t2
inner join zones on t2.dolocationid = zones.locationid
order by t2.total desc
