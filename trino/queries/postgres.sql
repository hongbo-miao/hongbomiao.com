select
    id,
    user_name,
    experiment_description,
    experiment_start_datetime,
    experiment_end_datetime,
    sessions,
    from_unixtime(cast(json_extract(sessions, '$.0.session_start_datetime') as double)) as session_start_datetime
from postgresql.public.metadata_motor
limit 100;

select
    t.id,
    t.user_name,
    t.experiment_description,
    t.experiment_start_datetime,
    t.experiment_end_datetime,
    s.session_description,
    from_unixtime(cast(s.session_start_datetime as double)) as session_start_datetime,
    case
        when s.session_end_datetime != '' then from_unixtime(cast(s.session_end_datetime as double))
    end as session_end_datetime
from postgresql.public.metadata_motor as t
cross join
    unnest(
        cast(
            json_extract(sessions, '$') as array (
                row(
                    session_description varchar,
                    session_start_datetime varchar,
                    session_end_datetime varchar
                )
            )
        )
    ) as s
where
    t.id = uuid '826db7a1-e01f-494d-b034-11f2b20949ce';
