create user mlflow_user with encrypted password 'xxx';

create database mlflow_db;
create database mlflow_auth_db;

grant all privileges on database mlflow_db to mlflow_user;
grant all privileges on database mlflow_auth_db to mlflow_user;

\connect mlflow_db
grant all privileges on schema public to mlflow_user;
\connect mlflow_auth_db
grant all privileges on schema public to mlflow_user;
