variable "common_tags" {
  type = map(string)
}
variable "amazon_msk_connector_name" {
  type = string
}
variable "kafka_topic" {
  type = string
}
variable "max_task_number" {
  type = number
}
variable "max_worker_number" {
  type = number
}
variable "worker_microcontroller_unit_number" {
  type = number
}
variable "snowflake_user_name" {
  type = string
}
variable "snowflake_private_key" {
  type = string
}
variable "snowflake_private_key_passphrase" {
  type = string
}
variable "snowflake_role_name" {
  type = string
}
variable "snowflake_database_name" {
  type = string
}
variable "snowflake_schema_name" {
  type = string
}
variable "confluent_schema_registry_url" {
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
variable "amazon_msk_cluster_bootstrap_servers" {
  type = string
}
variable "amazon_vpc_security_group_id" {
  type = string
}
variable "amazon_vpc_subnet_ids" {
  type = list(string)
}
variable "msk_log_s3_bucket_name" {
  type = string
}
variable "msk_log_s3_key" {
  type = string
}
