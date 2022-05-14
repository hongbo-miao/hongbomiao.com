FROM eqalpha/keydb:x86_64_v6.2.2

# Install libgomp1
# https://github.com/RedisGraph/RedisGraph/blob/master/README.md#loading-redisgraph-into-redis
RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get install libgomp1 -y --no-install-recommends \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

COPY ["hm-keydb/modules/", "/usr/lib/keydb/modules/"]
CMD ["keydb-server", \
  "/etc/keydb/keydb.conf", \
  "--loadmodule", "/usr/lib/keydb/modules/redisgraph.so"]
