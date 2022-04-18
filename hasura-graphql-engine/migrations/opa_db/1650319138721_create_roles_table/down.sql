drop trigger if exists update_updated_at on roles;
alter table roles replica identity default;
drop table if exists roles;
drop function if exists trigger_update_updated_at;
