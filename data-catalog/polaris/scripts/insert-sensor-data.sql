-- Create schema for sensor data
create schema if not exists iceberg.sensors;

-- Create sensor readings table
create table if not exists iceberg.sensors.motor (
    motor_id varchar,
    motor_temperature_c double,
    motor_rpm double,
    timestamp_ns bigint
)
with (
    format = 'parquet'
);

-- Insert fake sensor data
insert into iceberg.sensors.motor (
    motor_id,
    motor_temperature_c,
    motor_rpm,
    timestamp_ns
)
values
    ('motor_001', 65.2, 1475.0, 1734087600000000000),
    ('motor_001', 65.8, 1480.0, 1734088500000000000),
    ('motor_001', 66.4, 1488.0, 1734089400000000000),
    ('motor_002', 58.1, 1795.0, 1734087600000000000),
    ('motor_002', 58.6, 1802.0, 1734088500000000000),
    ('motor_002', 59.0, 1810.0, 1734089400000000000),
    ('motor_003', 72.8, 955.0, 1734087600000000000),
    ('motor_003', 73.4, 960.0, 1734088500000000000),
    ('motor_003', 74.1, 968.0, 1734089400000000000),
    ('motor_004', 49.3, 2905.0, 1734087600000000000),
    ('motor_004', 49.7, 2912.0, 1734088500000000000),
    ('motor_004', 50.2, 2920.0, 1734089400000000000),
    ('motor_005', 80.5, 1205.0, 1734087600000000000),
    ('motor_005', 81.1, 1210.0, 1734088500000000000),
    ('motor_005', 81.8, 1218.0, 1734089400000000000);

-- Verify data
select * from iceberg.sensors.motor order by timestamp_ns, motor_id;
