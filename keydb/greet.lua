local name = redis.call("get", KEYS[1])
local greet = ARGV[1]
local result = greet.." "..name
return result
