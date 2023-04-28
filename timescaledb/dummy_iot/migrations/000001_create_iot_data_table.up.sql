create table iot_data (
    timestamp timestamptz not null,
    temperature numeric not null,
    humidity numeric not null,
    primary key (timestamp)
);

select create_hypertable('iot_data', 'timestamp');
