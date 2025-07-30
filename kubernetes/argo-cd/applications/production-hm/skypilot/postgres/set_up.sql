create user skypilot_user with encrypted password 'xxx';

create database skypilot_db;

grant all privileges on database skypilot_db to skypilot_user;

\connect skypilot_db
grant all privileges on schema public to skypilot_user;
