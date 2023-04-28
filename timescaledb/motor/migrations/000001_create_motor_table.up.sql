create table motor (
    timestamp timestamptz not null,
    current double precision,
    voltage double precision,
    temperature double precision,
    primary key (timestamp)
);

select create_hypertable('motor', 'timestamp');
