FROM docker.io/eqalpha/keydb:x86_64_v6.2.2

# Install libgomp1
# https://github.com/RedisGraph/RedisGraph/blob/master/README.md#loading-redisgraph-into-redis
RUN apt-get update \
  && apt-get upgrade --yes \
  && apt-get install --yes --no-install-recommends libgomp1 \
  && apt-get clean \
  && rm -r -f /var/lib/apt/lists/*

COPY ["data-storage/hm-keydb/modules/", "/usr/lib/keydb/modules/"]
CMD ["keydb-server", \
  "/etc/keydb/keydb.conf", \
  "--loadmodule", "/usr/lib/keydb/modules/redisgraph.so"]
