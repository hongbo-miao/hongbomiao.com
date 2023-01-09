insert into iot_data (timestamp, temperature, humidity)
select
    ts,
    random() * 50 + 20 as temperature,
    random() * 50 + 50 as humidity
from (
    select generate_series(
        current_timestamp - interval '1 day',
        current_timestamp,
        interval '1 second'
    ) as ts
) as t;
