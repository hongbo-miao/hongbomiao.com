variable "name" {
  type = string
}
variable "proxy_to" {
  type = string
}
variable "active" {
  type = bool
}
variable "routing_group" {
  type = string
}
variable "external_url" {
  type    = string
  default = ""
}
