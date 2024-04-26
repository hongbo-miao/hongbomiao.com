show functions;

select * from system.runtime.nodes;

describe delta.production_hm_iot_db.motor;

analyze delta.production_hm_iot_db.motor;

show stats for delta.production_hm_iot_db.motor;
show stats for (select * from delta.production_hm_iot_db.motor where voltage > 10);

-- Show table data description language (DDL)
show create table delta.production_hm_iot_db.motor;
