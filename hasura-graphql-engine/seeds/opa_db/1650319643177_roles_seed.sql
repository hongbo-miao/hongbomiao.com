with ins (opal_client_name, role, allow) as (
    values
    (
        'hm-opal-client',
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
        ]'::json
    ),
    (
        'hm-opal-client',
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
        ]'::json
    ),
    (
        'hm-opal-client',
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
        ]'::json
    )
)

insert
into role (opal_client_id, role, allow)
select
    opal_client.id,
    ins.role,
    ins.allow
from opal_client
inner join ins on ins.opal_client_name = opal_client.name;
