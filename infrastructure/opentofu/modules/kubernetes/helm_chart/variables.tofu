variable "repository" {
  type = string
}
variable "chart_name" {
  type = string
}
variable "chart_version" {
  type = string
}
variable "release_name" {
  type = string
}
variable "my_values" {
  type    = list(string)
  default = []
}
variable "my_values_yaml_path" {
  type    = string
  default = ""
}
variable "namespace" {
  type = string
}
