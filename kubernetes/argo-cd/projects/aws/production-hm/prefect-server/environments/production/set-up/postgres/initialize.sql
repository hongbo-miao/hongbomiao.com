create user prefect_user with encrypted password 'xxx';

create database prefect_db;

grant all privileges on database prefect_db to prefect_user;

\connect prefect_db
grant all privileges on schema public to prefect_user;
