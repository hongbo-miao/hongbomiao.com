variable "project_id" {
  type = string
}
variable "network_id" {
  type = string
}
variable "admin_username" {
  type = string
}
variable "admin_password" {
  type      = string
  sensitive = true
}
variable "service_account_id" {
  type = string
}
variable "mlflow_cluster_name" {
  type = string
}
variable "mlflow_cluster_size" {
  type = string
}
variable "public_access" {
  type = bool
}
variable "storage_bucket_name" {
  type = string
}
variable "labels" {
  type = map(string)
}
