create or replace function trigger_update_updated_at()
  returns trigger as
$$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

create table if not exists opal_clients
(
  opal_client_id text        not null,
  config         text        not null,
  created_on     timestamptz not null default now(),
  updated_at     timestamptz not null default now(),
  primary key (opal_client_id)
);

create trigger update_updated_at
  before update
  on opal_clients
  for each row
execute procedure trigger_update_updated_at();
