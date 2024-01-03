FROM docker.io/rasa/rasa-sdk:3.7.1
WORKDIR /app

# COPY actions/requirements-actions.txt ./

USER root:root
# RUN pip install --requirement=requirements-actions.txt

COPY hm-rasa/actions /app/actions
USER 1001
