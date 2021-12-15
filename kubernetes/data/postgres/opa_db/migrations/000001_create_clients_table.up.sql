create or replace function trigger_update_updated_at()
  returns trigger as
$$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

create table if not exists clients
(
  client_id   uuid                 default gen_random_uuid(),
  client_name text        not null,
  created_on  timestamptz not null default now(),
  updated_at  timestamptz not null default now(),
  primary key (client_id)
);

create trigger update_updated_at
  before update
  on clients
  for each row
execute procedure trigger_update_updated_at();
