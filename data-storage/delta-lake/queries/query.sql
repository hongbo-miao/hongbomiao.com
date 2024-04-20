select
    timestamp,
    current,
    voltage,
    temperature
from delta.`s3a://hm-production-bucket/delta-tables/motor_data` limit 10;

select count(*) from delta.`s3a://hm-production-bucket/delta-tables/motor_data`;
