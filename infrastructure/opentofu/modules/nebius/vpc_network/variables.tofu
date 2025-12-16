variable "project_id" {
  type = string
}
variable "network_name" {
  type = string
}
variable "ipv4_private_pool_id" {
  type = string
}
variable "labels" {
  type    = map(string)
  default = {}
}
