variable "name" {
  type = string
}
variable "subjects" {
  type = list(string)
}
variable "storage" {
  type = string
}
variable "retention" {
  type = string
}
variable "max_message_count" {
  type = number
}
variable "max_age" {
  type = number
}
variable "max_bytes" {
  type = number
}
variable "discard" {
  type = string
}
variable "replica_count" {
  type = number
}
