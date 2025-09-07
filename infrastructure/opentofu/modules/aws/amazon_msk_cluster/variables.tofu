variable "common_tags" {
  type = map(string)
}
variable "amazon_msk_cluster_name" {
  type = string
}
variable "kafka_version" {
  type = string
}
variable "kafka_broker_number" {
  type = string
}
variable "kafka_broker_instance_type" {
  type = string
}
variable "kafka_broker_log_s3_bucket_name" {
  type = string
}
variable "amazon_ebs_volume_size_gb" {
  type = number
}
variable "amazon_vpc_security_group_id" {
  type = string
}
variable "amazon_vpc_subnet_ids" {
  type = list(string)
}
variable "aws_kms_key_arn" {
  type    = string
  default = null
}
variable "is_scram_enabled" {
  type    = bool
  default = false
}
