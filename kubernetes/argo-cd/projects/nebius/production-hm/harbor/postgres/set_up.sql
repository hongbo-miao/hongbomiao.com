create user harbor_user with encrypted password 'xxx';

create database harbor_db;

grant all privileges on database harbor_db to harbor_user;

\connect harbor_db
grant all privileges on schema public to harbor_user;
