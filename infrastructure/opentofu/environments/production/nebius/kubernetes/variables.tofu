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
variable "gatus_postgres_bootstrap_user_password" {
  type      = string
  sensitive = true
}
variable "grafana_postgres_bootstrap_user_password" {
  type      = string
  sensitive = true
}
variable "harbor_postgres_bootstrap_user_password" {
  type      = string
  sensitive = true
}
variable "mlflow_postgres_bootstrap_user_password" {
  type      = string
  sensitive = true
}
variable "polaris_postgres_bootstrap_user_password" {
  type      = string
  sensitive = true
}
variable "skypilot_postgres_bootstrap_user_password" {
  type      = string
  sensitive = true
}
variable "trino_gateway_postgres_bootstrap_user_password" {
  type      = string
  sensitive = true
}
