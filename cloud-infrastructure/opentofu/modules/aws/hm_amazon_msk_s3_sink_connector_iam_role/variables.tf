variable "common_tags" {
  type = map(string)
}
variable "amazon_msk_connector_name" {
  type = string
}
variable "amazon_msk_arn" {
  type = string
}
variable "msk_plugin_s3_bucket_name" {
  type = string
}
variable "msk_log_s3_bucket_name" {
  type = string
}
variable "msk_data_s3_bucket_name" {
  type = string
}
