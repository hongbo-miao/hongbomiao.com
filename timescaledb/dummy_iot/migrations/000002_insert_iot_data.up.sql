insert into iot_data (timestamp, temperature, humidity)
select
    ts,
    random() * 50.0 + 20.0 as temperature,
    random() * 50.0 + 50.0 as humidity
from (
    select
        generate_series(
            current_timestamp - interval '1 day',
            current_timestamp,
            interval '1 second'
        ) as ts
) as t;
