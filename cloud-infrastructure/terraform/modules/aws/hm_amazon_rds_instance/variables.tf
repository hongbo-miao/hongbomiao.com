variable "amazon_rds_name" {
  type = string
}
variable "amazon_rds_engine" {
  type = string
}
variable "amazon_rds_engine_version" {
  type = string
}
variable "amazon_rds_instance_class" {
  type = string
}
variable "storage_size_gb" {
  type = number
}
variable "max_storage_size_gb" {
  type = number
}
variable "user_name" {
  type = string
}
variable "password" {
  type = string
}
variable "vpc_security_group_ids" {
  type = list(string)
}
variable "subnet_group_name" {
  type = string
}
variable "parameter_group_name" {
  type = string
}
variable "cloudwatch_log_types" {
  type = list(string)
}
variable "environment" {
  type = string
}
variable "team" {
  type = string
}
