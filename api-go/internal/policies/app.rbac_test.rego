package app.rbac

user_roles = {
	"0x1": ["admin"],
	"0x2": [
		"employee",
		"billing",
	],
	"0x3": ["customer"],
}

role_grants = {
	"customer": [
		{
			"action": "read",
			"type": "dog",
		},
		{
			"action": "read",
			"type": "cat",
		},
		{
			"action": "adopt",
			"type": "dog",
		},
		{
			"action": "adopt",
			"type": "cat",
		},
	],
	"employee": [
		{
			"action": "read",
			"type": "dog",
		},
		{
			"action": "read",
			"type": "cat",
		},
		{
			"action": "update",
			"type": "dog",
		},
		{
			"action": "update",
			"type": "cat",
		},
	],
	"billing": [
		{
			"action": "read",
			"type": "finance",
		},
		{
			"action": "update",
			"type": "finance",
		},
	],
}

test_alice {
	allow with input as {"uid": "0x1", "action": "read", "type": "dog", "object": "id123"}
		 with data.user_roles as user_roles
		 with data.role_grants as role_grants

	allow with input as {"uid": "0x1", "action": "adopt", "type": "dog", "object": "id123"}
		 with data.user_roles as user_roles
		 with data.role_grants as role_grants
}

test_bob {
	allow with input as {"uid": "0x2", "action": "read", "type": "dog", "object": "id123"}
		 with data.user_roles as user_roles with data.role_grants as role_grants

	not allow with input as {"uid": "0x2", "action": "adopt", "type": "dog", "object": "id123"}
		 with data.user_roles as user_roles
		 with data.role_grants as role_grants
}
