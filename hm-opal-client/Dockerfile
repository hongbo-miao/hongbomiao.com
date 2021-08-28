# https://github.com/authorizon/opal/blob/master/docs/HOWTO/write_your_own_fetch_provider.md

FROM authorizon/opal-client:0.1.12
WORKDIR /usr/src/app

COPY ["hm-opal-client/", "./"]

RUN apk update \
  && apk add --no-cache gcc=9.3.0-r0 libc-dev=0.7.2-r0 \
  && python setup.py install \
  && apk del gcc libc-dev \
  && rm -rf /var/cache/apk
