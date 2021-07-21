package app.rbac

user_roles = {
	"alice": ["admin"],
	"bob": [
		"employee",
		"billing",
	],
	"eve": ["customer"],
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
	allow with input as {"user": "alice", "action": "read", "type": "dog", "object": "id123"}
		 with data.user_roles as user_roles
		 with data.role_grants as role_grants

	allow with input as {"user": "alice", "action": "adopt", "type": "dog", "object": "id123"}
		 with data.user_roles as user_roles
		 with data.role_grants as role_grants
}

test_bob {
	allow with input as {"user": "bob", "action": "read", "type": "dog", "object": "id123"}
		 with data.user_roles as user_roles with data.role_grants as role_grants

	not allow with input as {"user": "bob", "action": "adopt", "type": "dog", "object": "id123"}
		 with data.user_roles as user_roles
		 with data.role_grants as role_grants
}
