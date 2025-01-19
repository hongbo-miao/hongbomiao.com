aws-glue-databrew-dataset-list:
	aws databrew list-datasets
aws-glue-databrew-dataset-list:
	aws databrew list-datasets

aws-glue-databrew-recipe-create:
	aws databrew create-recipe --name=adsb-2x-flight-trace-recipe --steps=file://adsb-2x-flight-trace-recipe.json
aws-glue-databrew-recipe-update:
	aws databrew update-recipe --name=adsb-2x-flight-trace-recipe --steps=file://adsb-2x-flight-trace-recipe.json
aws-glue-databrew-recipe-publish:
	aws databrew publish-recipe --name=adsb-2x-flight-trace-recipe
aws-glue-databrew-recipe-list-versions:
	aws databrew list-recipe-versions --name=adsb-2x-flight-trace-recipe --no-cli-pager

aws-glue-databrew-job-list:
	aws databrew list-jobs
aws-glue-databrew-job-describe:
	aws databrew describe-job --name=hm-write-adsb-2x-flight-trace-csv-to-parquet-2023-10-01 --no-cli-pager
	aws databrew describe-job --name=hm-profile-adsb-2x-flight-trace-raw-parquet-2023-10-01 --no-cli-pager
aws-glue-databrew-job-describe:
	aws databrew start-job-run --name=hm-write-adsb-2x-flight-trace-csv-to-parquet-2023-10-01
	aws databrew start-job-run --name=hm-profile-adsb-2x-flight-trace-raw-parquet-2023-10-01
aws-glue-databrew-job-delete:
	aws databrew delete-job --name=hm-write-csv-to-parquet-adsb-2x-flight-trace-data

profile-adsb-2x-flight-trace-raw-parquet:
	bash bin/profile_adsb_2x_flight_trace_raw_parquet.sh
