create catalog paimon_catalog with (
    'type' = 'paimon',
    'warehouse' = 's3://paimon-warehouse/',
    's3.endpoint' = 'http://rustfs-svc.rustfs:9000',
    's3.access-key' = 'rustfs_admin',
    's3.secret-key' = 'passw0rd',
    's3.path.style.access' = 'true'
);

set 'execution.runtime-mode' = 'batch';
set 'sql-client.execution.result-mode' = 'tableau';

select count(*) as row_count from paimon_catalog.sensor_db.sensor_reading;
