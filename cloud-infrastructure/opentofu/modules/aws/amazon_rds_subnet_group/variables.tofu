variable "common_tags" {
  type = map(string)
}
variable "subnet_group_name" {
  type = string
}
variable "subnet_ids" {
  type = list(string)
}
