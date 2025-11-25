select case relreplident
    when 'd' then 'default'
    when 'n' then 'nothing'
    when 'f' then 'full'
    when 'i' then 'index'
    end as replica_identity
from pg_class
where oid = 'role'::regclass;
