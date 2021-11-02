insert into roles (role, allow)
values
  ('customer', '[
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
  ('employee', '[
    {
      "action": "read",
      "resource": "dog"
    },
    {
      "action": "read",
      "resource": "cat"
    }
  ]'::json),
  ('billing', '[
    {
      "action": "read",
      "resource": "finance"
    },
    {
      "action": "update",
      "resource": "finance"
    }
  ]'::json);
