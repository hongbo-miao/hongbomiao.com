insert into iot_data (timestamp, temperature, humidity)
select
    ts,
    avg(random() * 50 + 20) over (order by ts rows between 29 preceding and current row) as temperature,
    avg(random() * 50 + 50) over (order by ts rows between 29 preceding and current row) as humidity
from (
    select generate_series(
        current_timestamp - interval '2 year',
        current_timestamp,
        interval '1 minute'
    ) as ts
) as t;
