variable "grafana_admin_username" {
  type      = string
  sensitive = true
}
variable "grafana_admin_password" {
  type      = string
  sensitive = true
}
variable "managed_tag" {
  type = string
}
