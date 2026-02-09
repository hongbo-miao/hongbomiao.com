create user gatus_user with encrypted password 'xxx';

create database gatus_db;

grant all privileges on database gatus_db to gatus_user;

\connect gatus_db
grant all privileges on schema public to gatus_user;
