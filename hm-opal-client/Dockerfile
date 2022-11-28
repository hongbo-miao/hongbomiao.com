FROM docker.io/permitio/opal-client:0.3.1
WORKDIR /usr/src/app

COPY ["hm-opal-client/", "./"]
RUN pip install --no-cache-dir --requirement=requirements.txt
