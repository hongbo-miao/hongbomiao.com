create or replace function trigger_update_updated_at()
returns trigger as
$$
begin
    new.updated_at = now();
return new;
end;
$$ language plpgsql;

create table movies
(
    id uuid default gen_random_uuid(),
    title text not null,
    release_date date not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    primary key (id)
);

create or replace trigger update_updated_at
before update
on movies
for each row
execute procedure trigger_update_updated_at();
