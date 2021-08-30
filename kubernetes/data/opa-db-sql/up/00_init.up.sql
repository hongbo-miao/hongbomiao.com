create or replace function trigger_update_updated_at()
  returns trigger as
$$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

create table roles
(
  role_id    uuid                 default gen_random_uuid(),
  role       text        not null,
  allow      jsonb       not null,
  created_on timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  primary key (role_id)
);

create trigger update_updated_at
  before update
  on roles
  for each row
execute procedure trigger_update_updated_at();
