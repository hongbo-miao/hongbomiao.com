create user trino_gateway_user with encrypted password 'xxx';

create database trino_gateway_db;

grant all privileges on database trino_gateway_db to trino_gateway_user;

\connect trino_gateway_db
grant all privileges on schema public to trino_gateway_user;
