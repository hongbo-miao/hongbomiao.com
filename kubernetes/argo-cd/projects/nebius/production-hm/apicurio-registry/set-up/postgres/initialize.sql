create user apicurio_registry_user with encrypted password 'xxx';
create database apicurio_registry_db;
grant all privileges on database apicurio_registry_db to apicurio_registry_user;
\connect apicurio_registry_db
grant all privileges on schema public to apicurio_registry_user;
