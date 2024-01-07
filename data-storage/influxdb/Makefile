# InfluxDB
influx-install:
	brew install influxdb-cli

influx-config-list:
	influx config list
influx-config-create:
	influx config create \
		--org=primary \
		--config-name=hm-influxdb \
		--host-url=https://influxdb.hongbomiao.com
influx-config-delete:
	influx config rm hm-influxdb
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
	influx bucket list \
		--org=primary
influx-bucket-create:
	export INFLUX_TOKEN=xxx && \
	influx bucket create \
		--org=primary \
		--name=hm-bucket

influx-delete:
	export INFLUX_TOKEN=xxx && \
	influx delete \
		--org=primary \
		--bucket=hm-bucket \
		--start=2010-01-01T00:00:00Z \
		--stop=2030-01-01T00:00:00Z

# hm-grinding-wheel-station-bucket
influx-query-generate-grinding-wheel-station-data:
	export INFLUX_TOKEN=xxx && \
	influx query \
		--org=primary \
		--file=queries/hm-grinding-wheel-station-bucket/000001_generate_grinding_wheel_station_data.flux
influx-query-query-machinery:
	export INFLUX_TOKEN=xxx && \
	influx query \
		--org=primary \
		--file=queries/hm-grinding-wheel-station-bucket/000002_query_machinery.flux

# Chronograf
chronograf:
	docker run --publish=8888:8888 chronograf
