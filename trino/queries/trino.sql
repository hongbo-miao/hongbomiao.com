show functions;

describe delta.hm_iot_db.motor;

analyze delta.hm_iot_db.motor;

show stats for delta.hm_iot_db.motor;
show stats for (select * from delta.hm_iot_db.motor where voltage > 10);
