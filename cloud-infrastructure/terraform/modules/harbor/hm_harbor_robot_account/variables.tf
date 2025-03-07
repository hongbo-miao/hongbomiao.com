variable "name" {
  type = string
}
variable "secret" {
  type      = string
  sensitive = true
}
