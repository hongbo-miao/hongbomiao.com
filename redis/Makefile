# Redis
redis-start:
	redis-server
redis-stop:
	redis-cli SHUTDOWN
redis-cli:
	redis-cli
redis-cli-ping:
	redis-cli PING
redis-cli-keys:
	redis-cli KEYS "*"
redis-cli-set:
	redis-cli SET hm-key hongbo
redis-cli-get:
	redis-cli GET hm-key

# KeyDB
keydb-start:
	keydb-server
keydb-stop:
	keydb-cli SHUTDOWN
keydb-cli:
	keydb-cli
keydb-cli-ping:
	keydb-cli PING
redis-cli-keys:
	keydb-cli KEYS "*"
keydb-cli-set:
	keydb-cli SET hm-key hongbo
keydb-cli-get:
	keydb-cli GET hm-key

# KeyDB (multi-master)
keydb-start-multi-master:
	keydb-server --multi-master yes --active-replica yes --replicaof 192.168.2.58 6379 --requirepass passw0rd --masterauth passw0rd
	keydb-server --multi-master yes --active-replica yes --replicaof 192.168.2.117 6379 --requirepass passw0rd --masterauth passw0rd
keydb-cli-pass:
	keydb-cli --pass passw0rd
