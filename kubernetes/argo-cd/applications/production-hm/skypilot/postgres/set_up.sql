create user skypilot_user with encrypted password 'Hh3MRHMG9lcIlDDYgoE2tvbDWtHcCLm93Rys5RD2x3K0Mkie3H';

create database skypilot_db;

grant all privileges on database skypilot_db to skypilot_user;

\connect skypilot_db
grant all privileges on schema public to skypilot_user;
