create external table awsdatacatalog.hm_iot_db.motor
location 's3://hongbomiao-bucket/delta-tables/motor_data/'
tblproperties (
    'table_type' = 'delta'
);

select
    timestamp,
    current,
    voltage,
    temperature
from awsdatacatalog.hm_iot_db.motor limit 100;

show partitions awsdatacatalog.hm_iot_db.motor;
