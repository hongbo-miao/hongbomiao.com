# https://samples.adsbexchange.com/hires-traces/
aws-databrew-create-recipe:
	aws databrew create-recipe \
		--name=adsb-2x-flight-trace-recipe \
		--steps=file://adsb-2x-flight-trace-recipe.json
aws-databrew-update-recipe:
	aws databrew update-recipe \
		--name=adsb-2x-flight-trace-recipe \
		--steps=file://adsb-2x-flight-trace-recipe.json
aws-databrew-publish-recipe:
	aws databrew publish-recipe \
		--name=adsb-2x-flight-trace-recipe
