#!/usr/bin/env bash

database_name="adsb_db"
source="adsb_2x_flight_trace"

table_name="${source}_data"
echo "Creating delta table ${table_name}"
query="create external table awsdatacatalog.${database_name}.${table_name} location 's3://hm-production-bucket/data/delta-tables/${table_name}' tblproperties ('table_type' = 'delta')"
aws athena start-query-execution --work-group=hm-workgroup --query-execution-context=Database="${database_name}" --query-string="${query}" --no-cli-pager
sleep 1
aws glue create-partition-index --database-name="${database_name}" --table-name="${table_name}" --partition-index=Keys=_date,IndexName=_date_idx
