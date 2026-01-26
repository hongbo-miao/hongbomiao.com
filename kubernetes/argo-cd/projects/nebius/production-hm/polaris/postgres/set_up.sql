create user polaris_user with encrypted password 'xxx';

create database polaris_db;

grant all privileges on database polaris_db to polaris_user;

\connect polaris_db
grant all privileges on schema public to polaris_user;
