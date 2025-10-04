package app.rbac

# Define test roles
roles = {
	"billing": {
		"role": "billing",
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
	},
	"customer": {
		"role": "customer",
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
	},
	"employee": {
		"role": "employee",
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
	},
}

# Test admin role permissions
test_admin_permissions {
	# Admin should have access to all resources and actions
	allow with input as {"roles": ["admin"], "action": "read", "resource": "dog"}
		with data.roles as roles

	allow with input as {"roles": ["admin"], "action": "read", "resource": "cat"}
		with data.roles as roles

	allow with input as {"roles": ["admin"], "action": "adopt", "resource": "dog"}
		with data.roles as roles

	allow with input as {"roles": ["admin"], "action": "adopt", "resource": "cat"}
		with data.roles as roles

	allow with input as {"roles": ["admin"], "action": "read", "resource": "finance"}
		with data.roles as roles

	allow with input as {"roles": ["admin"], "action": "update", "resource": "finance"}
		with data.roles as roles
}

# Test employee role permissions
test_employee_permissions {
	# Employee should be able to read animals
	allow with input as {"roles": ["employee"], "action": "read", "resource": "dog"}
		with data.roles as roles

	allow with input as {"roles": ["employee"], "action": "read", "resource": "cat"}
		with data.roles as roles

	# Employee should not be able to adopt animals
	not allow with input as {"roles": ["employee"], "action": "adopt", "resource": "dog"}
		with data.roles as roles

	not allow with input as {"roles": ["employee"], "action": "adopt", "resource": "cat"}
		with data.roles as roles

	# Employee should not have access to finance
	not allow with input as {"roles": ["employee"], "action": "read", "resource": "finance"}
		with data.roles as roles

	not allow with input as {"roles": ["employee"], "action": "update", "resource": "finance"}
		with data.roles as roles
}

# Test customer role permissions
test_customer_permissions {
	# Customer should be able to read and adopt animals
	allow with input as {"roles": ["customer"], "action": "read", "resource": "dog"}
		with data.roles as roles

	allow with input as {"roles": ["customer"], "action": "read", "resource": "cat"}
		with data.roles as roles

	allow with input as {"roles": ["customer"], "action": "adopt", "resource": "dog"}
		with data.roles as roles

	allow with input as {"roles": ["customer"], "action": "adopt", "resource": "cat"}
		with data.roles as roles

	# Customer should not have access to finance
	not allow with input as {"roles": ["customer"], "action": "read", "resource": "finance"}
		with data.roles as roles

	not allow with input as {"roles": ["customer"], "action": "update", "resource": "finance"}
		with data.roles as roles
}

# Test billing role permissions
test_billing_permissions {
	# Billing should have access to finance
	allow with input as {"roles": ["billing"], "action": "read", "resource": "finance"}
		with data.roles as roles

	allow with input as {"roles": ["billing"], "action": "update", "resource": "finance"}
		with data.roles as roles

	# Billing should not have access to animals
	not allow with input as {"roles": ["billing"], "action": "read", "resource": "dog"}
		with data.roles as roles

	not allow with input as {"roles": ["billing"], "action": "read", "resource": "cat"}
		with data.roles as roles

	not allow with input as {"roles": ["billing"], "action": "adopt", "resource": "dog"}
		with data.roles as roles

	not allow with input as {"roles": ["billing"], "action": "adopt", "resource": "cat"}
		with data.roles as roles
}
