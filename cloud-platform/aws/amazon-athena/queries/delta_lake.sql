create external table awsdatacatalog.production_hm_iot_db.motor
location 's3://hm-production-bucket/delta-tables/motor_data/'
tblproperties (
    'table_type' = 'delta'
);

select
    timestamp,
    current,
    voltage,
    temperature
from awsdatacatalog.production_hm_iot_db.motor limit 100;

show partitions awsdatacatalog.production_hm_iot_db.motor;
