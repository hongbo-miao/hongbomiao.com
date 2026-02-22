package app.rbac

default allow := false

allow if {
	user_is_admin
}

allow if {
	some grant in user_is_granted
	input.action == grant.action
	input.resource == grant.resource
}

user_is_admin if {
	some role in input.roles
	role == "admin"
}

user_is_granted contains grant if {
	some role in input.roles
	some grant in data.roles[role].allow
}
