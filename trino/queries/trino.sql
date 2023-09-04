show functions;

describe delta.hm_iot_db.motor;

analyze delta.hm_iot_db.motor;

show stats for delta.hm_iot_db.motor;
show stats for (select * from delta.hm_iot_db.motor where voltage > 10);

-- Show table data description language (DDL)
show create table delta.hm_iot_db.motor;
