FROM alpine:3.15.4 AS builder
RUN mkdir -p /usr/lib/keydb/modules/ \
  && wget https://github.com/Hongbo-Miao/hm-redis-modules/raw/main/redisgraph.Linux-ubuntu18.04-x86_64.2.8.11/redisgraph.so

FROM eqalpha/keydb:x86_64_v6.2.2
# https://github.com/RedisGraph/RedisGraph/blob/master/README.md#loading-redisgraph-into-redis
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y libgomp1
COPY --from=builder /usr/lib/keydb/modules/ /usr/lib/keydb/modules/
CMD ["keydb-server", "/etc/keydb/keydb.conf", "--loadmodule", "/usr/lib/keydb/modules/redisgraph.so"]
