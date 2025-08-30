variable "common_tags" {
  type = map(string)
}
variable "parameter_group_name" {
  type = string
}
variable "family" {
  type = string
}
variable "parameters" {
  type = list(object({
    name  = string
    value = string
  }))
  default = []
}

