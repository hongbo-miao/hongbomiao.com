# InfluxDB
influx-install:
	brew install influxdb-cli

influx-config-list:
	influx config list
influx-config-create:
	influx config create --config-name=hm-influxdb --host-url=http://localhost:20622 --org=primary
influx-config-activate:
	influx config hm-influxdb

influx-ping:
	influx ping
influx-version:
	influx version

influx-org-list:
	export INFLUX_TOKEN=xxx && \
	influx org list

influx-bucket-list:
	export INFLUX_TOKEN=xxx && \
	influx bucket list --org=primary
influx-bucket-create:
	export INFLUX_TOKEN=xxx && \
	influx bucket create --name=hm-bucket --org=primary

# hm-grinding-wheel-station-bucket
influx-query-generate-grinding-wheel-station-data:
	export INFLUX_TOKEN=xxx && \
	influx query --file=queries/hm-grinding-wheel-station-bucket/000001_generate_grinding_wheel_station_data.flux --org=primary
influx-query-query-machinery:
	export INFLUX_TOKEN=xxx && \
	influx query --file=queries/hm-grinding-wheel-station-bucket/000002_query_machinery.flux --org=primary

# Chronograf
chronograf:
	docker run --publish=8888:8888 chronograf
