# Ansible Galaxy
ansible-galaxy-install:
	ansible-galaxy install --role-file=requirements.yml
ansible-galaxy-update:
	ansible-galaxy install --role-file=requirements.yml --force

ansible-galaxy-collection-list:
	ansible-galaxy collection list
ansible-galaxy-collection-install:
	ansible-galaxy collection install xxx
ansible-galaxy-collection-update:
	ansible-galaxy collection install xxx --force

ansible-galaxy-role-list:
	ansible-galaxy role list
ansible-galaxy-role-install:
	ansible-galaxy install xxx
ansible-galaxy-role-update:
	ansible-galaxy install xxx --force

# Ansible Vault
ansible-vault-encrypt-string:
	ansible-vault encrypt_string 'xxx'
ansible-vault-decrypt-string:
	echo '$ANSIBLE_VAULT;1.1;AES256 xxx' | tr -d ' ' | ansible-vault decrypt && echo

ansible-doc-list:
	ansible-doc --list

ansible-inventory-list:
	ansible-inventory --inventory=inventory.yaml --list
ansible-inventory-graph:
	ansible-inventory --inventory=inventory.yaml --graph

ansible-run-ping:
	ansible hm-ubuntu-group --inventory=inventory.yaml --vault-password-file=~/.ansible_vault_pass --module-name=ping
ansible-run-show-system-info:
	ansible hm-ubuntu-group --inventory=inventory.yaml --vault-password-file=~/.ansible_vault_pass --module-name=setup
ansible-run-uptime:
	ansible hm-ubuntu-group --inventory=inventory.yaml --vault-password-file=~/.ansible_vault_pass --args=uptime

ansible-playbook-list-tags:
	ansible-playbook hm-ubuntu-group/playbook.yml --list-tags

ansible-playbook-hm-macos-group:
	ansible-playbook --inventory=inventory.yaml hm-macos-group/playbook.yml
ansible-playbook-hm-ubuntu-group:
	ansible-playbook --inventory=inventory.yaml --vault-password-file=~/.ansible_vault_pass hm-ubuntu-group/playbook.yml
ansible-playbook-hm-windows-group:
	ansible-playbook --inventory=inventory.yaml --vault-password-file=~/.ansible_vault_pass hm-windows-group/playbook.yml
