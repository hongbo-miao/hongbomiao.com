package app.rbac

default allow = false

allow {
	user_is_admin
}

allow {
	some grant
	user_is_granted[grant]
	input.action == grant.action
	input.resource == grant.resource
}

user_is_admin {
	some i
	input.roles[i] == "admin"
}

user_is_granted[grant] {
	some i, j
	role := input.roles[i]
	grant := data.roles[role].allow[j]
}
