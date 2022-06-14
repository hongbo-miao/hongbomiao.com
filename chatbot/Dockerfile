FROM rasa/rasa-sdk:3.2.0
WORKDIR /app

# COPY actions/requirements-actions.txt ./

USER root:root
# RUN pip install -r requirements-actions.txt

COPY chatbot/actions /app/actions
USER 1001
