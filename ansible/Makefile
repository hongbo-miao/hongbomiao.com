ansible-galaxy-collection-list:
	ansible-galaxy collection list
ansible-galaxy-collection-install:
	ansible-galaxy collection install community.general
ansible-galaxy-role-list:
	ansible-galaxy role list
ansible-galaxy-role-install:
	ansible-galaxy install bartdorlandt.poetry
	ansible-galaxy install staticdev.pyenv
	ansible-galaxy install stefangweichinger.ansible_rclone

ansible-doc-list:
	ansible-doc --list

ansible-inventory-list:
	ansible-inventory --inventory=inventory.yaml --list
ansible-inventory-graph:
	ansible-inventory --inventory=inventory.yaml --graph

ansible-vault-encrypt-string:
	ansible-vault encrypt_string xxx

ansible-run-ping:
	ansible all --inventory=inventory.yaml --module-name=ping
ansible-run-uptime:
	ansible all --inventory=inventory.yaml --args=uptime

ansible-playbook-hm-macos-group:
	ansible-playbook --inventory=inventory.yaml hm-macos-group/playbook.yml
ansible-playbook-hm-ubuntu-group:
	ansible-playbook --inventory=inventory.yaml hm-ubuntu-group/playbook.yml --vault-password-file=~/.ansible_vault_pass
ansible-playbook-hm-windows-group:
	ansible-playbook --inventory=inventory.yaml hm-windows-group/playbook.yml --vault-password-file=~/.ansible_vault_pass
