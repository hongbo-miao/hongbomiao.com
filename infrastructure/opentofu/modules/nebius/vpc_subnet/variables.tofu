variable "project_id" {
  type = string
}
variable "network_id" {
  type = string
}
variable "subnet_name" {
  type = string
}
variable "private_ipv4_cidr_block" {
  type    = string
  default = null
}
variable "public_ipv4_cidr_block" {
  type    = string
  default = null
}
variable "labels" {
  type    = map(string)
  default = {}
}
