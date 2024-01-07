variable "amazon_emr_cluster_name" {
  type = string
}
variable "amazon_emr_version" {
  type = string
}
variable "applications" {
  type = list(string)
}
variable "primary_instance_type" {
  type = string
}
variable "core_instance_type" {
  type = string
}
variable "core_target_on_demand_capacity" {
  type = number
}
variable "bootstrap_set_up_script_s3_uri" {
  type = string
}
variable "configurations_json_string" {
  type = string
}
variable "steps" {
  type    = list(any)
  default = []
}
variable "iam_role_arn" {
  type = string
}
variable "environment" {
  type = string
}
variable "team" {
  type = string
}
