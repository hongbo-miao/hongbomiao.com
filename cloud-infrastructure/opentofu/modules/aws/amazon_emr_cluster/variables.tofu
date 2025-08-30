variable "common_tags" {
  type = map(string)
}
variable "amazon_emr_cluster_name" {
  type = string
}
variable "amazon_emr_version" {
  type = string
}
variable "applications" {
  type = list(string)
}
variable "primary_instance_target_on_demand_capacity" {
  type = number
}
variable "primary_instance_type" {
  type = string
}
variable "primary_instance_ebs_volume_size_gb" {
  type = string
}
variable "core_instance_target_on_demand_capacity" {
  type = number
}
variable "core_instance_weighted_capacity" {
  type = number
}
variable "core_instance_type" {
  type = string
}
variable "core_instance_ebs_volume_size_gb" {
  type = string
}
variable "bootstrap_set_up_script_s3_uri" {
  type = string
}
variable "configurations_json_string" {
  type = string
}
variable "placement_group_config" {
  type = list(object({
    instance_role      = string
    placement_strategy = string
  }))
  default = []
}
variable "steps" {
  type    = list(any)
  default = []
}
variable "iam_role_arn" {
  type = string
}
