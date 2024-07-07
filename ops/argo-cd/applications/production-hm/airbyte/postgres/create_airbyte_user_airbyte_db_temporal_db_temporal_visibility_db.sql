create user airbyte_user with encrypted password 'xxx';

create database airbyte_db;
create database temporal_db;
create database temporal_visibility_db;

grant all privileges on database airbyte_db to airbyte_user;
grant all privileges on database temporal_db to airbyte_user;
grant all privileges on database temporal_visibility_db to airbyte_user;

\connect airbyte_db
grant all privileges on schema public to airbyte_user;
\connect temporal_db
grant all privileges on schema public to airbyte_user;
\connect temporal_visibility_db
grant all privileges on schema public to airbyte_user;
