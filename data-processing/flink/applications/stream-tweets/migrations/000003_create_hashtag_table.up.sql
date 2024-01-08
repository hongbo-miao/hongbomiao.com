create table if not exists hashtag
(
    id uuid default gen_random_uuid(),
    name text not null,
    text text,
    primary key (id)
);
