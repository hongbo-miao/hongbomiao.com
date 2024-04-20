#!/usr/bin/env bash

database_name="iot_db"
source="motor"

for hex_digit in {0..9} {a..f}; do
  table_name="${source}_data_${hex_digit}"
  echo "Creating delta table ${table_name}"
  query="create external table awsdatacatalog.${database_name}.${table_name} location 's3://hm-production-bucket/data/delta-tables/${source}_data/${table_name}' tblproperties ('table_type' = 'delta')"
  aws athena start-query-execution --work-group=hm-workgroup --query-execution-context=Database="${database_name}" --query-string="${query}" --no-cli-pager
  sleep 1
  aws glue create-partition-index --database-name="${database_name}" --table-name="${table_name}" --partition-index=Keys=_event_id,IndexName=_event_id_idx
done
