-- https://docs.airbyte.com/integrations/sources/postgres
-- https://wolfman.dev/posts/pg-logical-heartbeats

create user airbyte_user password 'xxx';

grant usage on schema public to airbyte_user;
grant select on all tables in schema public to airbyte_user;
alter default privileges in schema public grant select on tables to airbyte_user;

grant rds_replication to airbyte_user;

select pg_create_logical_replication_slot('airbyte_public_logical_replication_slot', 'pgoutput');
-- List: select * from pg_replication_slots;
-- Drop: select pg_drop_replication_slot('xxx_logical_replication_slot');

do $$
    declare
        schema_name text := 'public';
        table_names text[] := array[
            'my_table_1',
            'my_table_2'
        ];
        table_name text;
    begin
        foreach table_name in array table_names
            loop
                execute format('
                alter table %1$s.%2$s replica identity default;
            ', schema_name, table_name);
            end loop;
    end $$;


create publication airbyte_public_publication for table
public.my_table_1,
public.my_table_2;
-- List: select * from pg_publication;
-- View: select * from pg_publication_tables where pubname = 'xxx_publication' order by schemaname, tablename;
-- Insert: alter publication xxx_publication add table my_schema.my_new_table;
-- Drop: drop publication xxx_publication;

create table if not exists public._airbyte_heartbeat (
    id serial primary key,
    timestamp timestamptz not null default now()
);
grant insert, update on table public._airbyte_heartbeat to airbyte_user;

-- In Amazon RDS, set max_slot_wal_keep_size to 524288 (512 GB)
