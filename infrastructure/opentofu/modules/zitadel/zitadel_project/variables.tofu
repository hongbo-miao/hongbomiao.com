variable "name" {
  type = string
}
variable "org_id" {
  type = string
}
variable "project_role_assertion" {
  type    = bool
  default = true
}
variable "project_role_check" {
  type    = bool
  default = false
}
variable "has_project_check" {
  type    = bool
  default = false
}
