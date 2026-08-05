create catalog paimon with (
    'type' = 'paimon',
    'warehouse' = 's3://paimon-warehouse/',
    's3.endpoint' = 'http://rustfs-svc:9000',
    's3.access-key' = 'rustfs_admin',
    's3.secret-key' = 'passw0rd',
    's3.path.style.access' = 'true'
);

set 'execution.runtime-mode' = 'batch';
set 'sql-client.execution.result-mode' = 'tableau';

-- Counts trail Postgres by up to one checkpoint interval, because Paimon only becomes
-- readable once a checkpoint commits.
select 'customers' as table_name, count(*) as row_count from paimon.customer_data.customers
union all
select 'orders' as table_name, count(*) as row_count from paimon.customer_data.orders;
