ansible-galaxy-collection-list:
	ansible-galaxy collection list
ansible-galaxy-role-list:
	ansible-galaxy role list

ansible-inventory-list:
	ansible-inventory --inventory=hosts --list

ansible-run-ping:
	ansible all --inventory=hosts --module-name=ping
ansible-run-uptime:
	ansible all --inventory=hosts --args=uptime

ansible-playbook-hm-macos:
	ansible-playbook --inventory=hosts hm-macos/playbook.yml
ansible-playbook-hm-ubuntu:
	ansible-playbook --inventory=hosts hm-ubuntu/playbook.yml --ask-become-pass
