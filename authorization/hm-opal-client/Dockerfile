FROM docker.io/permitio/opal-client:0.8.2
WORKDIR /usr/src/app

COPY ["authorization/hm-opal-client/requirements.txt", "./"]
RUN pip install --no-cache-dir --requirement=requirements.txt

COPY ["authorization/hm-opal-client/", "./"]
