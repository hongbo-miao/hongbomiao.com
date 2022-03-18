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
	redis-cli SET hm-name Hongbo
redis-cli-get:
	redis-cli GET hm-name

# Redis - Lua
redis-cli-lua:
	redis-cli EVAL 'return ARGV[1].." "..redis.call("GET",KEYS[1])' 1 hm-name "Hello"
redis-cli-lua-script-run:
	redis-cli --eval greet.lua hm-name , Hello

redis-cli-lua-script-load:
	redis-cli SCRIPT LOAD "$$(cat greet.lua)"
redis-cli-lua-script-run-hash:
	redis-cli EVALSHA c72b1fc2261c116957bbc839b558190a23d1cbb3 1 hm-name "Hello"
redis-cli-lua-script-exists:
	redis-cli SCRIPT EXISTS c72b1fc2261c116957bbc839b558190a23d1cbb3
redis-cli-lua-script-flush:
	redis-cli SCRIPT FLUSH

# KeyDB
keydb-start:
	keydb-server
keydb-stop:
	keydb-cli SHUTDOWN
keydb-cli:
	keydb-cli
keydb-cli-ping:
	keydb-cli PING
keydb-cli-keys:
	keydb-cli KEYS "*"
keydb-cli-set:
	keydb-cli SET hm-name Hongbo
keydb-cli-get:
	keydb-cli GET hm-name

# KeyDB (multi-master)
keydb-start-multi-master:
	keydb-server --multi-master yes --active-replica yes --replicaof 192.168.2.58 6379 --requirepass passw0rd --masterauth passw0rd
	keydb-server --multi-master yes --active-replica yes --replicaof 192.168.2.117 6379 --requirepass passw0rd --masterauth passw0rd
keydb-cli-pass:
	keydb-cli --pass passw0rd
keydb-cli-lua-script-load:
	keydb-cli --pass passw0rd SCRIPT LOAD "$$(cat greet.lua)"
