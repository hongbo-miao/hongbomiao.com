#!/bin/sh
set -e

# swanctl talks to charon over a Unix socket, so loading the connection config is a separate step
# from starting the daemon -- poll for the socket instead of guessing a fixed sleep.
/usr/sbin/charon-systemd &
charon_pid=$!

while [ ! -S /var/run/charon.vici ]; do
  sleep 0.5
done

swanctl --load-all

wait "${charon_pid}"
