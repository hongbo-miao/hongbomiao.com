select role, allow
from role
where opal_client_id = (
    select opal_client_id from opal_client where client_name = 'hm-opal-client'
);
