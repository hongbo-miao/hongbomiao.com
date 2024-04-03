# https://developer.hashicorp.com/terraform/language/resources/terraform-data
resource "terraform_data" "hm_local_tracker_sink_plugin" {
  triggers_replace = [
    var.snowflake_kafka_connector_version,
    var.bc_fips_version,
    var.bcpkix_fips_version
  ]
  provisioner "local-exec" {
    command = <<-EOF
      mkdir /tmp/tracker-sink-plugin/
      curl --fail --show-error --location --output /tmp/tracker-sink-plugin/snowflake-kafka-connector-${var.snowflake_kafka_connector_version}.jar https://repo1.maven.org/maven2/com/snowflake/snowflake-kafka-connector/${var.snowflake_kafka_connector_version}/snowflake-kafka-connector-${var.snowflake_kafka_connector_version}.jar
      curl --fail --show-error --location --output /tmp/tracker-sink-plugin/bc-fips-${var.bc_fips_version}.jar https://repo1.maven.org/maven2/org/bouncycastle/bc-fips/${var.bc_fips_version}/bc-fips-${var.bc_fips_version}.jar
      curl --fail --show-error --location --output /tmp/tracker-sink-plugin/bcpkix-fips-${var.bcpkix_fips_version}.jar https://repo1.maven.org/maven2/org/bouncycastle/bcpkix-fips/${var.bcpkix_fips_version}/bcpkix-fips-${var.bcpkix_fips_version}.jar
      zip --recurse-paths --junk-paths ${var.local_dir_path}/${var.local_file_name} /tmp/tracker-sink-plugin/
      rm -r -f /tmp/tracker-sink-plugin/
    EOF
  }
}
