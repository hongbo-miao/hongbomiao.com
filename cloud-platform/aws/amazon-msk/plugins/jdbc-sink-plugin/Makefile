plugin-build:
	bash build.sh
plugin-upload:
	aws s3 cp kafka-connect-jdbc.zip s3://hm-production-bucket/amazon-msk/plugins/jdbc-sink-plugin.zip
plugin-create:
	aws kafkaconnect create-custom-plugin --cli-input-json file://jdbc-sink-plugin.json
