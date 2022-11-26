create extension if not exists pgcrypto;

create table users
(
  id         uuid primary key     default gen_random_uuid(),
  email      text unique not null,
  password   text        not null,
  first_name text        not null,
  last_name  text        not null,
  bio        text,
  created_on timestamptz not null default now(),
  last_login timestamptz not null default now()
);
