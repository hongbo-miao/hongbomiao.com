variable "project_id" {
  type = string
}
variable "network_id" {
  type = string
}
variable "subnet_name" {
  type = string
}
variable "should_use_private_network_pools" {
  type    = bool
  default = false
}
variable "should_use_public_network_pools" {
  type    = bool
  default = false
}
variable "labels" {
  type    = map(string)
  default = {}
}
