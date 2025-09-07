variable "repository" {
  type = string
}
variable "ssh_private_key" {
  type      = string
  sensitive = true
}
