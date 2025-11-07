create table movies
(
    id uuid primary key default gen_random_uuid(),
    title text not null,
    release_date date not null
);
