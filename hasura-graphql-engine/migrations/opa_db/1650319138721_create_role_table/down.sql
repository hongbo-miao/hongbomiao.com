drop trigger if exists update_updated_at on role;
alter table role replica identity default;
drop table if exists role;
drop function if exists trigger_update_updated_at;
