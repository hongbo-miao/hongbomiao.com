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
