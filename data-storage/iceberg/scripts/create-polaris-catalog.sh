#!/bin/sh
set -e

POLARIS_HOST="${POLARIS_HOST:-polaris}"
POLARIS_PORT="${POLARIS_PORT:-8181}"
CLIENT_ID="${CLIENT_ID:-root}"
CLIENT_SECRET="${CLIENT_SECRET:-polaris_passw0rd}"
CATALOG_NAME="${CATALOG_NAME:-warehouse}"
S3_BUCKET="${S3_BUCKET:-warehouse}"
S3_ENDPOINT="${S3_ENDPOINT:-http://minio:9000}"
S3_ACCESS_KEY="${S3_ACCESS_KEY:-minio_admin}"
S3_SECRET_KEY="${S3_SECRET_KEY:-minio_passw0rd}"
S3_REGION="${S3_REGION:-us-west-2}"

echo "Waiting for Polaris to be ready..."
until curl -s "http://${POLARIS_HOST}:8182/q/health" > /dev/null 2>&1; do
  echo "Polaris not ready yet, waiting..."
  sleep 2
done
echo "Polaris is ready!"

echo "Getting OAuth2 token..."
TOKEN_RESPONSE=$(curl -s -X POST "http://${POLARIS_HOST}:${POLARIS_PORT}/api/catalog/v1/oauth/tokens" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials" \
  -d "client_id=${CLIENT_ID}" \
  -d "client_secret=${CLIENT_SECRET}" \
  -d "scope=PRINCIPAL_ROLE:ALL")

ACCESS_TOKEN=$(echo "${TOKEN_RESPONSE}" | sed -n 's/.*"access_token":"\([^"]*\)".*/\1/p')

if [ -z "${ACCESS_TOKEN}" ]; then
  echo "Failed to get access token. Response: ${TOKEN_RESPONSE}"
  exit 1
fi
echo "Got access token successfully!"

echo "Creating catalog '${CATALOG_NAME}'..."
CATALOG_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "http://${POLARIS_HOST}:${POLARIS_PORT}/api/management/v1/catalogs" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{
    \"name\": \"${CATALOG_NAME}\",
    \"type\": \"INTERNAL\",
    \"properties\": {
      \"default-base-location\": \"s3://${S3_BUCKET}\"
    },
    \"storageConfigInfo\": {
      \"storageType\": \"S3\",
      \"allowedLocations\": [\"s3://${S3_BUCKET}\"],
      \"endpointInternal\": \"${S3_ENDPOINT}\",
      \"pathStyleAccess\": true
    }
  }")

HTTP_CODE=$(echo "${CATALOG_RESPONSE}" | tail -n1)
RESPONSE_BODY=$(echo "${CATALOG_RESPONSE}" | sed '$d')

if [ "${HTTP_CODE}" = "201" ] || [ "${HTTP_CODE}" = "200" ]; then
  echo "Catalog '${CATALOG_NAME}' created successfully!"
elif [ "${HTTP_CODE}" = "409" ]; then
  echo "Catalog '${CATALOG_NAME}' already exists."
else
  echo "Failed to create catalog. HTTP ${HTTP_CODE}: ${RESPONSE_BODY}"
fi

echo "Granting CATALOG_MANAGE_CONTENT privilege to catalog_admin role..."
GRANT_RESPONSE=$(curl -s -w "\n%{http_code}" -X PUT "http://${POLARIS_HOST}:${POLARIS_PORT}/api/management/v1/catalogs/${CATALOG_NAME}/catalog-roles/catalog_admin/grants" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{
    \"type\": \"catalog\",
    \"privilege\": \"CATALOG_MANAGE_CONTENT\"
  }")

HTTP_CODE=$(echo "${GRANT_RESPONSE}" | tail -n1)
if [ "${HTTP_CODE}" = "201" ] || [ "${HTTP_CODE}" = "200" ]; then
  echo "Privilege granted successfully!"
else
  echo "Note: Could not grant privilege (may already exist or different API version)"
fi

echo "Polaris catalog setup complete!"
