package rbac.authz

# user-role assignments
user_roles := {
	"jack": ["qa", "engineer"],
	"rose": ["hr"],
}

# role-permissions assignments
role_permissions := {
	"qa": [{"action": "read", "object": "server123"}],
	"engineer": [
		{"action": "read", "object": "server123"},
		{"action": "write", "object": "server123"},
	],
	"hr": [{"action": "read", "object": "database456"}],
}

# logic that implements RBAC.
default allow = false

allow {
	# lookup the list of roles for the user
	roles := user_roles[input.user[_]]

	# for each role in that list
	r := roles[_]

	# lookup the permissions list for role r
	permissions := role_permissions[r]

	# for each permission
	p := permissions[_]

	# check if the permission granted to r matches the user's request
	p == {"action": input.action, "object": input.object}
}
