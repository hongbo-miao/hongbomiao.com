create or replace function trigger_update_updated_at()
  returns trigger as
$$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

create table if not exists roles
(
  id        uuid                 default gen_random_uuid(),
  opal_client_id uuid        not null,
  role           text        not null,
  allow          jsonb       not null,
  created_on     timestamptz not null default now(),
  updated_at     timestamptz not null default now(),
  primary key (id),
  constraint fk_space
    foreign key (opal_client_id)
      references opal_clients (id)
);

create trigger update_updated_at
  before update
  on roles
  for each row
execute procedure trigger_update_updated_at();
