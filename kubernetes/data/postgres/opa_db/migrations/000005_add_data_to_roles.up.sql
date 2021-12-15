with ins (opal_client_name, role, allow) as
       (values ('hm-opal-client',
                'customer',
                '[
                  {
                    "action": "read",
                    "resource": "dog"
                  },
                  {
                    "action": "read",
                    "resource": "cat"
                  },
                  {
                    "action": "adopt",
                    "resource": "dog"
                  },
                  {
                    "action": "adopt",
                    "resource": "cat"
                  }
                ]'::json),
               ('hm-opal-client',
                'employee',
                '[
                  {
                    "action": "read",
                    "resource": "dog"
                  },
                  {
                    "action": "read",
                    "resource": "cat"
                  }
                ]'::json),
               ('hm-opal-client',
                'billing',
                '[
                  {
                    "action": "read",
                    "resource": "finance"
                  },
                  {
                    "action": "update",
                    "resource": "finance"
                  }
                ]'::json)
       )
insert
into roles (opal_client_id, role, allow)
select opal_clients.id, ins.role, ins.allow
from opal_clients
       join ins on ins.opal_client_name = opal_clients.name;
