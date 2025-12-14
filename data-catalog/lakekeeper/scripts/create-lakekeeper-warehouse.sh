#!/bin/sh
set -e

LAKEKEEPER_HOST="${LAKEKEEPER_HOST:-lakekeeper}"
LAKEKEEPER_PORT="${LAKEKEEPER_PORT:-8181}"
WAREHOUSE_NAME="${WAREHOUSE_NAME:-warehouse}"
PROJECT_ID="${PROJECT_ID:-00000000-0000-0000-0000-000000000000}"
S3_BUCKET="${S3_BUCKET:-warehouse}"
S3_ENDPOINT="${S3_ENDPOINT:-http://minio:9000}"
S3_ACCESS_KEY="${S3_ACCESS_KEY:-minio_admin}"
S3_SECRET_KEY="${S3_SECRET_KEY:-minio_passw0rd}"
S3_REGION="${S3_REGION:-us-west-2}"

echo "Waiting for Lakekeeper to be ready..."
until curl --silent "http://${LAKEKEEPER_HOST}:${LAKEKEEPER_PORT}/health" > /dev/null 2>&1; do
  echo "Lakekeeper not ready yet, waiting..."
  sleep 2
done
echo "Lakekeeper is ready!"

echo "Creating warehouse '${WAREHOUSE_NAME}'..."
WAREHOUSE_RESPONSE=$(curl --silent --write-out "\n%{http_code}" --request POST "http://${LAKEKEEPER_HOST}:${LAKEKEEPER_PORT}/management/v1/warehouse" \
  --header "Content-Type: application/json" \
  --data "{
    \"warehouse-name\": \"${WAREHOUSE_NAME}\",
    \"project-id\": \"${PROJECT_ID}\",
    \"storage-profile\": {
      \"type\": \"s3\",
      \"bucket\": \"${S3_BUCKET}\",
      \"key-prefix\": \"\",
      \"assume-role-arn\": null,
      \"endpoint\": \"${S3_ENDPOINT}\",
      \"region\": \"${S3_REGION}\",
      \"path-style-access\": true,
      \"flavor\": \"minio\",
      \"sts-enabled\": true
    },
    \"storage-credential\": {
      \"type\": \"s3\",
      \"credential-type\": \"access-key\",
      \"aws-access-key-id\": \"${S3_ACCESS_KEY}\",
      \"aws-secret-access-key\": \"${S3_SECRET_KEY}\"
    }
  }")

HTTP_CODE=$(echo "${WAREHOUSE_RESPONSE}" | tail -n1)
RESPONSE_BODY=$(echo "${WAREHOUSE_RESPONSE}" | sed '$d')

if [ "${HTTP_CODE}" = "201" ] || [ "${HTTP_CODE}" = "200" ]; then
  echo "Warehouse '${WAREHOUSE_NAME}' created successfully!"
elif [ "${HTTP_CODE}" = "409" ]; then
  echo "Warehouse '${WAREHOUSE_NAME}' already exists."
else
  echo "Failed to create warehouse. HTTP ${HTTP_CODE}: ${RESPONSE_BODY}"
fi

echo "Lakekeeper warehouse setup complete!"
