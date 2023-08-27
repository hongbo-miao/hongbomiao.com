create-checkpoint-directory:
	mkdir /tmp/vector/
remove-checkpoint-directory:
	rm -r -f /tmp/vector/

generate-dummy-log:
	sh bin/generate-dummy-log.sh

vector:
	vector --config=vector.toml --openssl-legacy-provider=false
