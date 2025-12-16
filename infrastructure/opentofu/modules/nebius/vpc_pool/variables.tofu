variable "project_id" {
  type = string
}
variable "pool_name" {
  type = string
}
variable "ipv4_cidr_block" {
  type = string
}
variable "visibility" {
  type = string
}
variable "labels" {
  type    = map(string)
  default = {}
}
