FROM docker.io/permitio/opal-client:0.9.4
USER root
WORKDIR /usr/src/app

COPY ["security/hm-opal-client/requirements.txt", "./"]
RUN pip install --no-cache-dir --requirement=requirements.txt

COPY ["security/hm-opal-client/", "./"]
USER opal
