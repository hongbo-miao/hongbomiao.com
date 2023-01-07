FROM docker.io/prefecthq/prefect:2.7.7-python3.11
WORKDIR /usr/src/app

COPY ["hm-prefect/src/print_platform/requirements.txt", "./"]
RUN pip install --no-cache-dir --requirement=requirements.txt

COPY ["hm-prefect/src/print_platform/", "/opt/prefect/flows/"]
