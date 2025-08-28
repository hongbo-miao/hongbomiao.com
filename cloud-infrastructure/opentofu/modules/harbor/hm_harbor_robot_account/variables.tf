variable "name" {
  type = string
}
variable "project_names" {
  type = list(string)
}
variable "actions" {
  type = list(string)
}
variable "secret" {
  type      = string
  sensitive = true
}
