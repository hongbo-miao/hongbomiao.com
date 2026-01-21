variable "project_id" {
  type = string
}
variable "network_id" {
  type = string
}
variable "postgres_version" {
  type = string
}
variable "postgres_cluster_name" {
  type = string
}
variable "bootstrap_database_name" {
  type = string
}
variable "bootstrap_user_name" {
  type = string
}
variable "bootstrap_user_password" {
  type = string
}
variable "public_access" {
  type = bool
}
variable "pooling_mode" {
  type = string
}
variable "disk_size_gib" {
  type = number
}
variable "host_count" {
  type = number
}
variable "platform" {
  type = string
}
variable "preset" {
  type = string
}
variable "labels" {
  type = map(string)
}
variable "backup_window_start_time_utc" {
  type = string
}
variable "backup_retention_policy" {
  type = string
}
