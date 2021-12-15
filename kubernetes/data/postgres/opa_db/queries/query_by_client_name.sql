select role, allow
from roles
where client_id = (
  select client_id from clients where client_name = 'hm-opal-client'
);
