# https://developer.hashicorp.com/terraform/language/resources/terraform-data
resource "terraform_data" "hm_local_tracker_sink_plugin" {
  triggers_replace = [
    var.snowflake_kafka_connector_version,
    var.bc_fips_version,
    var.bcpkix_fips_version,
    var.confluent_kafka_connect_avro_converter_version
  ]
  provisioner "local-exec" {
    command = <<-EOF
      mkdir /tmp/${var.kafka_plugin_name}/

      curl --fail --show-error --location --output /tmp/${var.kafka_plugin_name}/snowflake-kafka-connector-${var.snowflake_kafka_connector_version}.jar https://repo1.maven.org/maven2/com/snowflake/snowflake-kafka-connector/${var.snowflake_kafka_connector_version}/snowflake-kafka-connector-${var.snowflake_kafka_connector_version}.jar
      curl --fail --show-error --location --output /tmp/${var.kafka_plugin_name}/bc-fips-${var.bc_fips_version}.jar https://repo1.maven.org/maven2/org/bouncycastle/bc-fips/${var.bc_fips_version}/bc-fips-${var.bc_fips_version}.jar
      curl --fail --show-error --location --output /tmp/${var.kafka_plugin_name}/bcpkix-fips-${var.bcpkix_fips_version}.jar https://repo1.maven.org/maven2/org/bouncycastle/bcpkix-fips/${var.bcpkix_fips_version}/bcpkix-fips-${var.bcpkix_fips_version}.jar

      curl --fail --show-error --location --output /tmp/confluentinc-kafka-connect-avro-converter-${var.confluent_kafka_connect_avro_converter_version}.zip https://d1i4a15mxbxib1.cloudfront.net/api/plugins/confluentinc/kafka-connect-avro-converter/versions/${var.confluent_kafka_connect_avro_converter_version}/confluentinc-kafka-connect-avro-converter-${var.confluent_kafka_connect_avro_converter_version}.zip
      unzip /tmp/confluentinc-kafka-connect-avro-converter-${var.confluent_kafka_connect_avro_converter_version}.zip -d /tmp/
      cp -R /tmp/confluentinc-kafka-connect-avro-converter-${var.confluent_kafka_connect_avro_converter_version}/lib/. /tmp/${var.kafka_plugin_name}/

      zip --recurse-paths --junk-paths ${var.local_dir_path}/${var.local_file_name} /tmp/${var.kafka_plugin_name}/
      rm -r -f /tmp/${var.kafka_plugin_name}/
    EOF
  }
}
