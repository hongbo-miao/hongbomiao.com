create user nimtable_user with encrypted password 'xxx';
create database nimtable_db;
grant all privileges on database nimtable_db to nimtable_user;
\connect nimtable_db
grant all privileges on schema public to nimtable_user;
