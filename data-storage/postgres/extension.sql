-- List extensions
select * from pg_available_extensions;

-- Install extension
create extension if not exists pg_graphql;
