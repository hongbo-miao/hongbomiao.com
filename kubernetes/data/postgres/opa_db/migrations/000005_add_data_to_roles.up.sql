with ins (client_name, role, allow) as
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
into roles (client_id, role, allow)
select clients.client_id, ins.role, ins.allow
from clients
       join ins on ins.client_name = clients.client_name;
