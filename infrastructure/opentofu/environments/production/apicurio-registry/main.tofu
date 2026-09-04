module "telemetry_value" {
  source      = "../../../modules/apicurio-registry/artifact"
  group_id    = "default"
  artifact_id = "telemetry-value"
  type        = "AVRO"
  content     = file("${path.module}/files/telemetry.avsc")
}
