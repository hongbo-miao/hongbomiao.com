create user grafana_user with encrypted password 'xxx';

create database grafana_db;

grant all privileges on database grafana_db to grafana_user;

\connect grafana_db
grant all privileges on schema public to grafana_user;
