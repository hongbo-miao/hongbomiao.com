variable "amazon_emr_cluster_name" {
  type = string
}
variable "amazon_emr_version" {
  type = string
}
variable "primary_instance_type" {
  type = string
}
variable "core_instance_type" {
  type = string
}
variable "core_instance_count" {
  type = number
}
variable "bootstrap_set_up_script_s3_uri" {
  type = string
}
variable "aws_iam_role" {
  type = string
}
variable "environment" {
  type = string
}
variable "team" {
  type = string
}
