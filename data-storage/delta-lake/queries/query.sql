select
    timestamp,
    current,
    voltage,
    temperature
from delta.`s3a://hm-production-bucket/delta-tables/motor_data`
order by timestamp desc
limit 100;

select count(*) from delta.`s3a://hm-production-bucket/delta-tables/motor_data`;
