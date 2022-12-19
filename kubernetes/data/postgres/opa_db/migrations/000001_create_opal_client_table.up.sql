create or replace function trigger_update_updated_at()
returns trigger as
$$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

create table if not exists opal_client
(
    id uuid default gen_random_uuid(),
    name text not null,
    config text,
    created_on timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    primary key (id)
);

create trigger update_updated_at
before update
on opal_client
for each row
execute procedure trigger_update_updated_at();
