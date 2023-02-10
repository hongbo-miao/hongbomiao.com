FROM docker.io/prefecthq/prefect:2.8.0-python3.11

COPY ["hm-prefect/src/print_platform/requirements.txt", "./"]
RUN pip install --no-cache-dir --requirement=requirements.txt

COPY ["hm-prefect/src/print_platform/", "/opt/prefect/flows/src/print_platform/"]
