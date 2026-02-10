variable "parent_id" {
  type = string
}
variable "name" {
  type = string
}
variable "min_node_count" {
  type = number
}
variable "max_node_count" {
  type = number
}
variable "platform" {
  type = string
}
variable "preset" {
  type = string
}
variable "os" {
  type = string
}
variable "labels" {
  type = map(string)
}
variable "cloud_init_user_data" {
  type    = string
  default = ""
}
variable "boot_disk_type" {
  type = string
}
variable "boot_disk_size_gibibytes" {
  type = number
}
