FROM docker.io/rasa/rasa-sdk:3.5.0
WORKDIR /app

# COPY actions/requirements-actions.txt ./

USER root:root
# RUN pip install --requirement=requirements-actions.txt

COPY chatbot/actions /app/actions
USER 1001
