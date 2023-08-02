ansible-galaxy-collection-list:
	ansible-galaxy collection list
ansible-galaxy-role-list:
	ansible-galaxy role list

ansible-inventory-list:
	ansible-inventory --inventory=inventory.yaml --list

ansible-run-ping:
	ansible all --inventory=inventory.yaml --module-name=ping
ansible-run-uptime:
	ansible all --inventory=inventory.yaml --args=uptime

ansible-playbook-hm-macos-group:
	ansible-playbook --inventory=inventory.yaml hm-macos-group/playbook.yml
ansible-playbook-hm-ubuntu-group:
	ansible-playbook --inventory=inventory.yaml hm-ubuntu-group/playbook.yml --ask-become-pass
