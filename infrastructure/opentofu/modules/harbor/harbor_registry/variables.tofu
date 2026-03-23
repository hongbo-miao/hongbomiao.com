variable "name" {
  type = string
}
variable "provider_name" {
  type = string
}
variable "endpoint_url" {
  type = string
}
variable "access_id" {
  type    = string
  default = null
}
variable "access_secret" {
  type      = string
  sensitive = true
  default   = null
}
