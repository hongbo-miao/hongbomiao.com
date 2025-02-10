FROM docker.io/confluentinc/cp-schema-registry:7.8.1

USER root

ADD --chown=appuser:appuser https://github.com/aws/aws-msk-iam-auth/releases/download/v2.2.0/aws-msk-iam-auth-2.2.0-all.jar /usr/share/java/cp-base-new/
ADD --chown=appuser:appuser https://github.com/aws/aws-msk-iam-auth/releases/download/v2.2.0/aws-msk-iam-auth-2.2.0-all.jar /usr/share/java/schema-registry/

ENV SCHEMA_REGISTRY_KAFKASTORE_SECURITY_PROTOCOL="SASL_SSL"
ENV SCHEMA_REGISTRY_KAFKASTORE_SASL_MECHANISM="AWS_MSK_IAM"
ENV SCHEMA_REGISTRY_KAFKASTORE_SASL_JAAS_CONFIG="software.amazon.msk.auth.iam.IAMLoginModule required;"
ENV SCHEMA_REGISTRY_KAFKASTORE_SASL_CLIENT_CALLBACK_HANDLER_CLASS="software.amazon.msk.auth.iam.IAMClientCallbackHandler"

USER appuser
