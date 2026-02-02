variable "trino_gateway_user_name" {
  type = string
}
variable "trino_gateway_password" {
  type      = string
  sensitive = true
}
