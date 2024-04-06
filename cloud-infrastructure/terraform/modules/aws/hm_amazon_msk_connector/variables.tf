variable "amazon_msk_connector_name" {
  type = string
}
variable "kafka_connect_version" {
  type = string
}
variable "amazon_msk_plugin_arn" {
  type = string
}
variable "amazon_msk_plugin_revision" {
  type = string
}
variable "amazon_msk_connector_iam_role_arn" {
  type = string
}
variable "snowflake_username" {
  type = string
}
variable "snowflake_private_key" {
  type = string
}
variable "snowflake_private_key_passphrase" {
  type = string
}
variable "msk_log_s3_bucket_name" {
  type = string
}
variable "msk_log_s3_key" {
  type = string
}
variable "environment" {
  type = string
}
variable "team" {
  type = string
}
