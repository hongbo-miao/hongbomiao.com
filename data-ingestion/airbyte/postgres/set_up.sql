create user airbyte_user password 'xxx';
grant usage on schema public to airbyte_user;
grant select on all tables in schema public to airbyte_user;
alter default privileges in schema public grant select on tables to airbyte_user;

-- In Amazon RDS, use `grant rds_replication to airbyte_user;` instead
alter user airbyte_user replication;

select pg_create_logical_replication_slot('airbyte_slot', 'pgoutput');

alter table inertial_navigation_system replica identity default;
alter table motor replica identity default;
create publication airbyte_publication for table inertial_navigation_system, motor;
