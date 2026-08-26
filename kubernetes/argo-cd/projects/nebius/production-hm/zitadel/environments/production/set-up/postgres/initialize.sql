create user zitadel_user with encrypted password 'xxx';
create database zitadel_db;
grant all privileges on database zitadel_db to zitadel_user;
\connect zitadel_db
grant all privileges on schema public to zitadel_user;
