select role, allow
from roles
where opal_client_id = (
  select opal_client_id from clients where client_name = 'hm-opal-client'
);
