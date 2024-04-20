variable "amazon_emr_cluster_id" {
  type = string
}
variable "task_instance_target_spot_capacity" {
  type = number
}
variable "task_instance_configs" {
  type = list(object({
    instance_type     = string
    weighted_capacity = number
  }))
}
