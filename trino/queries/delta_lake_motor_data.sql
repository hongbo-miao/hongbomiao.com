select
    timestamp,
    current,
    voltage,
    temperature
from delta.production_hm_iot_db.motor limit 100;

-- Get full data
with
t0 as (select * from delta.production_hm_delta_db.motor_data_0 where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
t1 as (select * from delta.production_hm_delta_db.motor_data_1 where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
t2 as (select * from delta.production_hm_delta_db.motor_data_2 where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
t3 as (select * from delta.production_hm_delta_db.motor_data_3 where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
t4 as (select * from delta.production_hm_delta_db.motor_data_4 where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
t5 as (select * from delta.production_hm_delta_db.motor_data_5 where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
t6 as (select * from delta.production_hm_delta_db.motor_data_6 where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
t7 as (select * from delta.production_hm_delta_db.motor_data_7 where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
t8 as (select * from delta.production_hm_delta_db.motor_data_8 where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
t9 as (select * from delta.production_hm_delta_db.motor_data_9 where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
ta as (select * from delta.production_hm_delta_db.motor_data_a where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
tb as (select * from delta.production_hm_delta_db.motor_data_b where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
tc as (select * from delta.production_hm_delta_db.motor_data_c where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
td as (select * from delta.production_hm_delta_db.motor_data_d where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
te as (select * from delta.production_hm_delta_db.motor_data_e where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
tf as (select * from delta.production_hm_delta_db.motor_data_f where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255')
select from_unixtime_nanos(_time) as _time, current, voltage, temperature
from t0
join t1 on t0._time = t1._time
join t2 on t0._time = t2._time
join t3 on t0._time = t3._time
join t4 on t0._time = t4._time
join t5 on t0._time = t5._time
join t6 on t0._time = t6._time
join t7 on t0._time = t7._time
join t8 on t0._time = t8._time
join t9 on t0._time = t9._time
join ta on t0._time = ta._time
join tb on t0._time = tb._time
join tc on t0._time = tc._time
join td on t0._time = td._time
join te on t0._time = te._time
join tf on t0._time = tf._time
order by t_0._time asc;

-- Get downsampled data
with
t_sec as (
    select max(_time) as _time
    from delta.production_hm_delta_db.motor_data_0
    group by date_trunc('second', from_unixtime_nanos(_time))
),
t0 as (select * from delta.production_hm_delta_db.motor_data_0 where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
t1 as (select * from delta.production_hm_delta_db.motor_data_1 where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
t2 as (select * from delta.production_hm_delta_db.motor_data_2 where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
t3 as (select * from delta.production_hm_delta_db.motor_data_3 where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
t4 as (select * from delta.production_hm_delta_db.motor_data_4 where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
t5 as (select * from delta.production_hm_delta_db.motor_data_5 where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
t6 as (select * from delta.production_hm_delta_db.motor_data_6 where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
t7 as (select * from delta.production_hm_delta_db.motor_data_7 where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
t8 as (select * from delta.production_hm_delta_db.motor_data_8 where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
t9 as (select * from delta.production_hm_delta_db.motor_data_9 where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
ta as (select * from delta.production_hm_delta_db.motor_data_a where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
tb as (select * from delta.production_hm_delta_db.motor_data_b where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
tc as (select * from delta.production_hm_delta_db.motor_data_c where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
td as (select * from delta.production_hm_delta_db.motor_data_d where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
te as (select * from delta.production_hm_delta_db.motor_data_e where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255'),
tf as (select * from delta.production_hm_delta_db.motor_data_f where _event_id = 'ad7953cd-6d49-4929-8180-99555bebc255')
select from_unixtime_nanos(_time) as _time, current, voltage, temperature
from t_sec
join t0 on t_sec._time = t0._time
join t1 on t_sec._time = t1._time
join t2 on t_sec._time = t2._time
join t3 on t_sec._time = t3._time
join t4 on t_sec._time = t4._time
join t5 on t_sec._time = t5._time
join t6 on t_sec._time = t6._time
join t7 on t_sec._time = t7._time
join t8 on t_sec._time = t8._time
join t9 on t_sec._time = t9._time
join ta on t_sec._time = ta._time
join tb on t_sec._time = tb._time
join tc on t_sec._time = tc._time
join td on t_sec._time = td._time
join te on t_sec._time = te._time
join tf on t_sec._time = tf._time
order by t_sec._time asc;
