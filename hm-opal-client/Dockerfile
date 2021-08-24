# https://github.com/authorizon/opal/blob/master/docs/HOWTO/write_your_own_fetch_provider.md

FROM authorizon/opal-client:0.1.12
WORKDIR /usr/src/app

COPY ["hm-opal-client/", "./"]
RUN python setup.py install
