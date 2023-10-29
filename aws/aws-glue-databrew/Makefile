aws-glue-databrew-create-recipe:
	aws databrew create-recipe --name=adsb-2x-flight-trace-recipe --steps=file://adsb-2x-flight-trace-recipe.json
aws-glue-databrew-update-recipe:
	aws databrew update-recipe --name=adsb-2x-flight-trace-recipe --steps=file://adsb-2x-flight-trace-recipe.json
aws-glue-databrew-publish-recipe:
	aws databrew publish-recipe --name=adsb-2x-flight-trace-recipe

aws-glue-databrew-delete-job:
	aws databrew delete-job --name=hm-write-csv-to-parquet-adsb-2x-flight-trace-data
