package rbac.authz

test_jack {
	allow with input as {"user": ["jack"], "action": "read", "object": "server123"}
	allow with input as {"user": ["jack"], "action": "write", "object": "server123"}
	not allow with input as {"user": ["jack"], "action": "read", "object": "database456"}
}

test_rose {
	allow with input as {"user": ["rose"], "action": "read", "object": "database456"}
	not allow with input as {"user": ["rose"], "action": "read", "object": "server123"}
	not allow with input as {"user": ["rose"], "action": "write", "object": "server123"}
}
