variable "org_id" {
  type = string
}
variable "project_id" {
  type = string
}
variable "name" {
  type = string
}
variable "redirect_uris" {
  type = list(string)
}
variable "post_logout_redirect_uris" {
  type = list(string)
}
variable "dev_mode" {
  type    = bool
  default = false
}
