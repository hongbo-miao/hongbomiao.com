variable "common_tags" {
  type = map(string)
}
variable "amazon_msk_connector_name" {
  type = string
}
variable "kafka_topics" {
  type = list(string)
}
variable "aws_region" {
  type = string
}
variable "s3_bucket_name" {
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
