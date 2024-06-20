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
variable "environment" {
  type = string
}
variable "team" {
  type = string
}
