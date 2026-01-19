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
variable "harbor_postgres_bootstrap_user_password" {
  type      = string
  sensitive = true
}
variable "mlflow_postgres_bootstrap_user_password" {
  type      = string
  sensitive = true
}
