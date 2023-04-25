#!/usr/bin/env bash
set -e

echo "# Install TimescaleDB"
# Follow "Install TimescaleDB" in timescaledb/bin/ubuntu/install_timescaledb.sh
echo "=================================================="

echo "# Create database hm_sensor_db"
kubectl port-forward service/timescale --namespace=hm-timescale 25495:5432 &
sleep 5
psql postgresql://admin:passw0rd@localhost:25495/postgres --command="create database hm_sensor_db;"
psql postgresql://admin:passw0rd@localhost:25495/postgres --command="grant all privileges on database hm_sensor_db to admin;"
echo "=================================================="

echo "# Install hm-kafka-sensor-kafka-connect"
kubectl apply --filename=kubernetes/manifests/hm-kafka/sensor-kafka-connect
# kubectl delete --filename=kubernetes/manifests/hm-kafka/sensor-kafka-connect
echo "=================================================="
