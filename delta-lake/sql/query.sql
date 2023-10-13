select
    timestamp,
    current,
    voltage,
    temperature
from delta.`s3a://hongbomiao-bucket/delta-tables/motor` limit 10;

select count(*) from delta.`s3a://hongbomiao-bucket/delta-tables/motor`;
