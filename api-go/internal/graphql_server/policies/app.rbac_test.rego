package app.rbac

roles = {
	"billing": {
		"allow": [
			{
				"action": "read",
				"resource": "finance",
			},
			{
				"action": "update",
				"resource": "finance",
			},
		],
		"role": "billing",
	},
	"customer": {
		"allow": [
			{
				"action": "read",
				"resource": "dog",
			},
			{
				"action": "read",
				"resource": "cat",
			},
			{
				"action": "adopt",
				"resource": "dog",
			},
			{
				"action": "adopt",
				"resource": "cat",
			},
		],
		"role": "customer",
	},
	"employee": {
		"allow": [
			{
				"action": "read",
				"resource": "dog",
			},
			{
				"action": "read",
				"resource": "cat",
			},
		],
		"role": "employee",
	},
}

test_alice {
	allow with input as {"roles": ["admin"], "action": "read", "resource": "dog"}
		 with data.roles as roles
		 with data.roles as roles

	allow with input as {"roles": ["admin"], "action": "adopt", "resource": "dog"}
		 with data.roles as roles
		 with data.roles as roles
}

test_bob {
	allow with input as {"roles": ["employee", "billing"], "action": "read", "resource": "dog"}
		 with data.roles as roles with data.roles as roles

	not allow with input as {"roles": ["employee", "billing"], "action": "adopt", "resource": "dog"}
		 with data.roles as roles
		 with data.roles as roles
}
