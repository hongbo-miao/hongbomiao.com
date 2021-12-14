#!/usr/bin/env bash
set -e

CLUSTERS=("$@")

echo "# Clean cluster certificates"
rm -f kubernetes/certificates/ca.crt
rm -f kubernetes/certificates/ca.key
for cluster in "${CLUSTERS[@]}"; do
  rm -f "kubernetes/certificates/${cluster}-issuer.crt"
  rm -f "kubernetes/certificates/${cluster}-issuer.key"
done
echo "=================================================="

echo "# Generate cluster certificates"
CA_DIR="kubernetes/certificates"

step certificate create "identity.linkerd.${ORG_DOMAIN}" \
  "${CA_DIR}/ca.crt" "${CA_DIR}/ca.key" \
  --profile=root-ca \
  --no-password \
  --insecure

for cluster in "${CLUSTERS[@]}"; do
  domain="${cluster}.${ORG_DOMAIN}"
  crt="${CA_DIR}/${cluster}-issuer.crt"
  key="${CA_DIR}/${cluster}-issuer.key"

  step certificate create "identity.linkerd.${domain}" "${crt}" "${key}" \
    --ca="${CA_DIR}/ca.crt" \
    --ca-key="${CA_DIR}/ca.key" \
    --profile=intermediate-ca \
    --not-after=8760h \
    --no-password \
    --insecure
done
echo "=================================================="

for cluster in "${CLUSTERS[@]}"; do
  echo "# Check Linkerd installation environment on: k3d-${cluster}"
  while ! linkerd check --context="k3d-${cluster}" --pre ; do :; done
  echo "=================================================="
done

for cluster in "${CLUSTERS[@]}"; do
  domain="${cluster}.${ORG_DOMAIN}"
  crt="${CA_DIR}/${cluster}-issuer.crt"
  key="${CA_DIR}/${cluster}-issuer.key"

  echo "# Install Linkerd on: k3d-${cluster}"
  linkerd install \
    --cluster-domain="${domain}" \
    --identity-trust-domain="${domain}" \
    --identity-trust-anchors-file="${CA_DIR}/ca.crt" \
    --identity-issuer-certificate-file="${crt}" \
    --identity-issuer-key-file="${key}" \
    --disable-heartbeat | \
    kubectl apply --context="k3d-${cluster}" --filename=-
  echo "=================================================="
done
sleep 30

for cluster in "${CLUSTERS[@]}"; do
  echo "# Check Linkerd on: k3d-${cluster}"
  while ! linkerd check --context="k3d-${cluster}" ; do :; done
  echo "=================================================="
done
