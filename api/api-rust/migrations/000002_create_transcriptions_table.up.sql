create or replace function trigger_update_updated_at()
returns trigger as
$$
begin
    new.updated_at = now();
return new;
end;
$$ language plpgsql;

create table if not exists transcriptions (
    id uuid default gen_random_uuid(),
    stream_id text not null,
    timestamp_ns bigint not null,
    text text not null,
    language text not null,
    duration_s double precision not null,
    segment_start_s double precision not null,
    segment_end_s double precision not null,
    words jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    primary key (id),
    constraint unique_stream_timestamp unique (stream_id, timestamp_ns)
);

create index if not exists transcriptions_stream_id_idx on transcriptions (stream_id);
create index if not exists transcriptions_timestamp_ns_idx on transcriptions (timestamp_ns);

create or replace trigger update_updated_at
before update
on transcriptions
for each row
execute procedure trigger_update_updated_at();
