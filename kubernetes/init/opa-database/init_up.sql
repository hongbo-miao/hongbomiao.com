create or replace function trigger_update_updated_at()
  returns trigger as
$$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

create table opa
(
  id         uuid primary key     default gen_random_uuid(),
  org        text        not null,
  data       jsonb       not null,
  created_on timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create trigger update_updated_at
  before update
  on opa
  for each row
execute procedure trigger_update_updated_at();
