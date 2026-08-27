FROM docker.io/ubuntu:24.04

# charon-systemd runs the strongSwan IKE daemon in the foreground, logging to stderr -- the one
# entry point that works in a container with no init system. libstrongswan-standard-plugins is
# included only to silence the wall of "plugin ... failed to load" warnings that
# --no-install-recommends otherwise produces; the tunnel negotiates fine without it, but the noise
# would mislead someone reading the logs to learn.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
      charon-systemd \
      iproute2 \
      iputils-ping \
      libstrongswan-standard-plugins \
      strongswan-swanctl \
      tcpdump \
    && rm --recursive --force /var/lib/apt/lists/*
