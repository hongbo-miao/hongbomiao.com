#!/usr/bin/env bash
set -e

echo "# Install TimescaleDB"
# Follow "Install TimescaleDB" in timescaledb/bin/ubuntu/install_timescaledb.sh
echo "=================================================="

echo "# Create database production_hm_iot_db"
kubectl port-forward service/timescale --namespace=hm-timescale 25495:5432 &
sleep 5
psql postgresql://admin:passw0rd@localhost:25495/postgres --command="create database production_hm_iot_db;"
psql postgresql://admin:passw0rd@localhost:25495/postgres --command="grant all privileges on database production_hm_iot_db to admin;"
echo "=================================================="

echo "# Migrate database production_hm_iot_db"
TIMESCALEDB_URL="postgresql://admin:passw0rd@localhost:25495/production_hm_iot_db?sslmode=disable&search_path=public"
migrate -database "${TIMESCALEDB_URL}" -path data-storage/timescaledb/motor/migrations up
echo "=================================================="

echo "# Create secret hm-iot-db-credentials"
kubectl create secret generic hm-iot-db-credentials \
  --from-file=kubernetes/manifests/hm-kafka/iot-kafka-connect/hm-motor-jdbc-sink-kafka-connector/iot-db-credentials.properties \
  --namespace=hm-kafka
# kubectl delete secret hm-iot-db-credentials --namespace=hm-kafka
echo "=================================================="

echo "# Install hm-kafka-iot-kafka-connect"
kubectl apply --filename=kubernetes/manifests/hm-kafka/iot-kafka-connect
# kubectl delete --filename=kubernetes/manifests/hm-kafka/iot-kafka-connect
echo "=================================================="
