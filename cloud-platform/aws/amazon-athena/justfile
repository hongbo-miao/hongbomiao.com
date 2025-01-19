amazon-athena-list-databases:
	aws athena list-databases \
		--catalog-name=AwsDataCatalog
amazon-athena-list-databases:
	aws athena list-table-metadata \
		--catalog-name=AwsDataCatalog \
		--database-name=production_hm_iot_db
amazon-athena-query:
	aws athena start-query-execution \
		--query-string='select * from hm_iot_delta_db.motor limit 10;' \
		--result-configuration='OutputLocation=s3://hm-production-bucket/amazon-athena/query-results/aws-cli/query-result/'
