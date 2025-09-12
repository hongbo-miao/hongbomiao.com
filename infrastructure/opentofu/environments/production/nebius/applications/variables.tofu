variable "environment" {
  type = string
}
variable "tenant_id" {
  type = string
}
variable "project_id" {
  type = string
}
variable "vpc_network_id" {
  type = string
}
variable "vpc_subnet_id" {
  type = string
}
variable "common_labels" {
  type = map(string)
}
variable "hm_mlflow_admin_password" {
  type      = string
  sensitive = true
}
