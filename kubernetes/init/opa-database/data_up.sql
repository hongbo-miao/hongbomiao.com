insert into opa (org, data)
values ('hm', '{
  "user_roles": {
    "0x1": [
      "admin"
    ],
    "0x2": [
      "employee",
      "billing"
    ],
    "0x3": [
      "customer"
    ]
  },
  "role_grants": {
    "customer": [
      {
        "action": "read",
        "type": "dog"
      },
      {
        "action": "read",
        "type": "cat"
      },
      {
        "action": "adopt",
        "type": "dog"
      },
      {
        "action": "adopt",
        "type": "cat"
      }
    ],
    "employee": [
      {
        "action": "read",
        "type": "dog"
      },
      {
        "action": "read",
        "type": "cat"
      },
      {
        "action": "update",
        "type": "dog"
      },
      {
        "action": "update",
        "type": "cat"
      }
    ],
    "billing": [
      {
        "action": "read",
        "type": "finance"
      },
      {
        "action": "update",
        "type": "finance"
      }
    ]
  }
}
'::json);
